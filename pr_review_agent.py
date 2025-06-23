import os
import re
import hmac
import hashlib
import asyncio
import requests
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from github import Github, GithubException
import base64
import json
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime

# Load environment variables
load_dotenv()

# FastAPI app for webhook server
app = FastAPI(title="GitHub PR Review Agent", version="1.0.0")

# State Schema - This defines what data flows through our graph
class PRReviewState(TypedDict):
    """
    State for GitHub PR Review workflow
    """
    pr_url: str
    repo_owner: str
    repo_name: str
    pr_number: int
    pr_data: Optional[dict]
    files_changed: List[dict]
    review_comments: List[dict]
    final_summary: Optional[str]
    github_comments_posted: bool
    webhook_event: Optional[dict]
    error: Optional[str]

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize GitHub client
github_client = Github(os.getenv("GITHUB_TOKEN"))

def verify_webhook_signature(payload_body: bytes, signature_header: str, secret: str) -> bool:
    """
    Verify GitHub webhook signature for security
    """
    if not signature_header:
        return False
    
    hash_object = hmac.new(
        secret.encode('utf-8'),
        msg=payload_body,
        digestmod=hashlib.sha256
    )
    expected_signature = "sha256=" + hash_object.hexdigest()
    
    return hmac.compare_digest(expected_signature, signature_header)

@app.post("/webhook/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    GitHub webhook endpoint to receive PR events
    """
    try:
        # Get raw payload
        payload_body = await request.body()
        signature = request.headers.get("X-Hub-Signature-256")
        event_type = request.headers.get("X-GitHub-Event")
        
        # Verify webhook signature if secret is configured
        webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET")
        if webhook_secret and not verify_webhook_signature(payload_body, signature, webhook_secret):
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        # Check if event type is supported
        if event_type != "pull_request":
            raise HTTPException(status_code=400, detail=f"Unsupported event type: {event_type}")
        
        # Parse JSON payload
        payload = json.loads(payload_body.decode('utf-8'))
        
        print(f"📨 Received GitHub webhook: {event_type}")
        
        # Handle pull request events
        if event_type == "pull_request":
            action = payload.get("action")
            pr_data = payload.get("pull_request", {})
            
            # Trigger review for opened, reopened, or synchronized PRs
            if action in ["opened", "reopened", "synchronize"]:
                pr_url = pr_data.get("html_url")
                pr_number = pr_data.get("number")
                repo_name = payload.get("repository", {}).get("full_name")
                
                print(f"🚀 Triggering AI review for PR #{pr_number}: {pr_url}")
                
                # Run review in background to respond quickly to webhook
                background_tasks.add_task(
                    process_webhook_pr_review,
                    pr_url,
                    payload
                )
                
                return JSONResponse({
                    "status": "success",
                    "message": f"AI review triggered for PR #{pr_number}",
                    "pr_url": pr_url
                })
            else:
                return JSONResponse({
                    "status": "ignored",
                    "message": f"PR action '{action}' does not trigger review"
                })
        
        return JSONResponse({
            "status": "ignored",
            "message": f"Event type '{event_type}' not handled"
        })
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except Exception as e:
        print(f"❌ Webhook error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def process_webhook_pr_review(pr_url: str, webhook_payload: dict):
    """
    Process PR review triggered by webhook
    """
    try:
        print(f"🔄 Processing webhook PR review for: {pr_url}")
        result = review_pr_from_webhook(pr_url, webhook_payload)
        
        if result.get("error"):
            print(f"❌ Webhook review failed: {result['error']}")
        else:
            print(f"✅ Webhook review completed successfully")
            
    except Exception as e:
        print(f"❌ Error processing webhook PR review: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "GitHub PR Review Agent"
    })

@app.get("/")
async def root():
    """
    Root endpoint with service info
    """
    return JSONResponse({
        "service": "GitHub PR Review Agent",
        "version": "1.0.0",
        "endpoints": {
            "webhook": "/webhook/github",
            "health": "/health"
        },
        "github_api_status": check_github_access_status()
    })

def check_github_access_status() -> dict:
    """
    Check GitHub API access status
    """
    try:
        user = github_client.get_user()
        rate_limit = github_client.get_rate_limit()
        return {
            "authenticated": True,
            "user": user.login,
            "rate_limit_remaining": rate_limit.core.remaining,
            "rate_limit_total": rate_limit.core.limit
        }
    except Exception as e:
        return {
            "authenticated": False,
            "error": str(e)
        }

def parse_pr_url(pr_url: str) -> tuple:
    """
    Parse GitHub PR URL to extract owner, repo, and PR number
    """
    pattern = r'https://github\.com/([^/]+)/([^/]+)/pull/(\d+)'
    match = re.match(pattern, pr_url)
    
    if not match:
        raise ValueError(f"Invalid GitHub PR URL format: {pr_url}")
    
    return match.group(1), match.group(2), int(match.group(3))

def fetch_pr_data(state: PRReviewState) -> PRReviewState:
    """
    Node to fetch real PR data from GitHub API
    """
    print("📥 Fetching PR data from GitHub...")
    
    try:
        # Parse PR URL
        repo_owner, repo_name, pr_number = parse_pr_url(state["pr_url"])
        
        # Get repository and PR
        repo = github_client.get_repo(f"{repo_owner}/{repo_name}")
        pr = repo.get_pull(pr_number)
        
        # Fetch PR files
        files = list(pr.get_files())
        
        # Process files and get their content/diffs
        files_changed = []
        for file in files:
            file_data = {
                "filename": file.filename,
                "status": file.status,
                "additions": file.additions,
                "deletions": file.deletions,
                "changes": file.changes,
                "patch": file.patch if file.patch else "",
                "blob_url": file.blob_url,
                "raw_url": file.raw_url
            }
            
            # For new files or small files, also get the full content
            if file.status in ['added', 'modified'] and file.additions < 500:
                try:
                    if file.status == 'added':
                        # For new files, get content from the PR branch
                        file_content = repo.get_contents(file.filename, ref=pr.head.sha)
                        if hasattr(file_content, 'content'):
                            content = base64.b64decode(file_content.content).decode('utf-8')
                            file_data["full_content"] = content[:2000]  # Limit content size
                except Exception as e:
                    print(f"⚠️  Could not fetch full content for {file.filename}: {str(e)}")
                    file_data["full_content"] = None
            
            files_changed.append(file_data)
        
        # Get PR metadata
        pr_data = {
            "title": pr.title,
            "description": pr.body or "",
            "author": pr.user.login,
            "state": pr.state,
            "created_at": pr.created_at.isoformat(),
            "updated_at": pr.updated_at.isoformat(),
            "base_branch": pr.base.ref,
            "head_branch": pr.head.ref,
            "mergeable": pr.mergeable,
            "additions": pr.additions,
            "deletions": pr.deletions,
            "changed_files": pr.changed_files,
            "commits": pr.commits,
            "labels": [label.name for label in pr.labels],
            "assignees": [assignee.login for assignee in pr.assignees],
            "requested_reviewers": [reviewer.login for reviewer in pr.requested_reviewers]
        }
        
        print(f"✅ Successfully fetched PR #{pr_number}: {pr.title}")
        print(f"📊 Files changed: {len(files_changed)}, +{pr.additions}/-{pr.deletions}")
        
        return {
            **state,
            "repo_owner": repo_owner,
            "repo_name": repo_name,
            "pr_number": pr_number,
            "pr_data": pr_data,
            "files_changed": files_changed
        }
        
    except GithubException as e:
        error_msg = f"GitHub API error: {e.data.get('message', str(e)) if hasattr(e, 'data') else str(e)}"
        print(f"❌ {error_msg}")
        return {**state, "error": error_msg}
    
    except ValueError as e:
        error_msg = f"URL parsing error: {str(e)}"
        print(f"❌ {error_msg}")
        return {**state, "error": error_msg}
    
    except Exception as e:
        error_msg = f"Unexpected error fetching PR data: {str(e)}"
        print(f"❌ {error_msg}")
        return {**state, "error": error_msg}

def analyze_code_changes(state: PRReviewState) -> PRReviewState:
    """
    Node to analyze the code changes in the PR using AI
    """
    print("🔍 Analyzing code changes with AI...")
    
    if not state.get("files_changed"):
        return {**state, "error": "No files to analyze"}
    
    review_comments = []
    pr_context = state["pr_data"]
    
    # Create context about the PR
    pr_context_text = f"""
    PR Title: {pr_context['title']}
    Description: {pr_context['description'][:500]}...
    Author: {pr_context['author']}
    Base Branch: {pr_context['base_branch']} → Head Branch: {pr_context['head_branch']}
    Total Changes: +{pr_context['additions']}/-{pr_context['deletions']} across {pr_context['changed_files']} files
    """
    
    for file_change in state["files_changed"]:
        print(f"  📄 Analyzing {file_change['filename']}...")
        
        # Skip binary files or very large files
        if not file_change.get("patch") and file_change["changes"] > 100:
            review_comments.append({
                "file": file_change['filename'],
                "comment": "⚠️ Large file or binary file - manual review recommended",
                "type": "info",
                "line_number": None,
                "github_comment": "⚠️ **Large file detected** - This file requires manual review due to its size."
            })
            continue
        
        # Determine file type for specialized analysis
        file_ext = os.path.splitext(file_change['filename'])[1].lower()
        language_context = get_language_context(file_ext)
        
        # Create a comprehensive prompt for code review
        system_prompt = f"""You are a senior software engineer conducting a thorough code review. 
        Provide constructive, actionable feedback focusing on:

        🔒 SECURITY: Look for vulnerabilities, input validation, authentication issues
        🚀 PERFORMANCE: Identify bottlenecks, inefficient algorithms, memory leaks
        🧹 CODE QUALITY: Check for best practices, readability, maintainability
        🧪 TESTING: Assess test coverage and quality
        📚 DOCUMENTATION: Evaluate comments and documentation needs
        🏗️ ARCHITECTURE: Review design patterns and structure
        
        {language_context}
        
        Be specific about line numbers when possible. Provide both positive feedback and areas for improvement.
        Format your response for GitHub comments with clear sections and actionable suggestions."""
        
        human_prompt = f"""
        PR Context:
        {pr_context_text}
        
        File Analysis:
        📁 File: {file_change['filename']}
        📈 Status: {file_change['status']}
        📊 Changes: +{file_change['additions']}/-{file_change['deletions']}
        
        Code Changes:
        ```diff
        {file_change.get('patch', 'No diff available')[:3000]}
        ```
        
        {f"Full File Content (first 2000 chars):\n```\n{file_change.get('full_content', '')}\n```" if file_change.get('full_content') else ""}
        
        Please provide a detailed code review for this file optimized for GitHub comments.
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = llm.invoke(messages)
            
            # Create GitHub-optimized comment
            github_comment = format_comment_for_github(response.content, file_change)
            
            review_comments.append({
                "file": file_change['filename'],
                "comment": response.content,
                "github_comment": github_comment,
                "type": "ai_review",
                "line_number": None,
                "file_status": file_change['status'],
                "changes_count": file_change['changes']
            })
            
        except Exception as e:
            review_comments.append({
                "file": file_change['filename'],
                "comment": f"❌ Error analyzing file: {str(e)}",
                "github_comment": f"❌ **Analysis Error**: Could not analyze this file due to: {str(e)}",
                "type": "error",
                "line_number": None
            })
    
    print(f"✅ Analyzed {len(review_comments)} files")
    return {
        **state,
        "review_comments": review_comments
    }

def format_comment_for_github(ai_comment: str, file_data: dict) -> str:
    """
    Format AI comment for GitHub with proper markdown and emojis
    """
    file_name = file_data['filename']
    file_status = file_data['status']
    changes = f"+{file_data['additions']}/-{file_data['deletions']}"
    
    github_comment = f"""## 🤖 AI Code Review for `{file_name}`

**File Status:** `{file_status}` | **Changes:** `{changes}`

{ai_comment}

---
*Generated by AI PR Review Agent* 🚀
"""
    return github_comment

def get_language_context(file_ext: str) -> str:
    """
    Get language-specific review context
    """
    language_contexts = {
        '.py': """
        Python-specific focus:
        - PEP 8 compliance and pythonic code
        - Proper exception handling
        - Security issues (SQL injection, XSS, etc.)
        - Performance (list comprehensions, generators)
        - Type hints and documentation strings
        """,
        '.js': """
        JavaScript-specific focus:
        - ES6+ best practices
        - Async/await usage
        - Memory leaks and closures
        - Security (XSS, prototype pollution)
        - Performance and bundle size impact
        """,
        '.tsx': """
        React/TypeScript-specific focus:
        - Component design patterns
        - Hook usage and dependencies
        - Type safety and interfaces
        - Performance (re-renders, memoization)
        - Accessibility compliance
        """,
        '.java': """
        Java-specific focus:
        - Object-oriented design principles
        - Exception handling patterns
        - Memory management
        - Concurrency and thread safety
        - Security and input validation
        """,
        '.go': """
        Go-specific focus:
        - Idiomatic Go patterns
        - Error handling
        - Goroutine and channel usage
        - Performance and memory allocation
        - Interface design
        """
    }
    
    return language_contexts.get(file_ext, "General code review focusing on best practices and security.")

def generate_summary(state: PRReviewState) -> PRReviewState:
    """
    Node to generate a comprehensive PR review summary
    """
    print("📝 Generating comprehensive review summary...")
    
    if not state.get("review_comments"):
        return {**state, "error": "No review comments to summarize"}
    
    pr_data = state["pr_data"]
    
    # Categorize comments by severity
    critical_issues = []
    improvements = []
    positive_feedback = []
    
    for comment in state["review_comments"]:
        content = comment["comment"].lower()
        if any(keyword in content for keyword in ["security", "vulnerability", "critical", "error", "bug"]):
            critical_issues.append(comment)
        elif any(keyword in content for keyword in ["good", "well", "excellent", "nice"]):
            positive_feedback.append(comment)
        else:
            improvements.append(comment)
    
    # Compile all review comments
    all_comments = "\n\n".join([
        f"**{comment['file']} ({comment.get('file_status', 'unknown')}):**\n{comment['comment']}" 
        for comment in state["review_comments"] if comment['type'] != 'error'
    ])
    
    summary_prompt = f"""
    You are a technical lead providing a comprehensive PR review summary.
    
    PR Details:
    - Title: {pr_data['title']}
    - Author: {pr_data['author']}
    - Changes: +{pr_data['additions']}/-{pr_data['deletions']} across {pr_data['changed_files']} files
    - Branch: {pr_data['base_branch']} ← {pr_data['head_branch']}
    
    Analysis Summary:
    - Critical Issues Found: {len(critical_issues)}
    - Improvement Suggestions: {len(improvements)}
    - Positive Aspects: {len(positive_feedback)}
    
    Create a professional PR review summary with:
    
    ## 🎯 Executive Summary
    - Overall assessment (Approve/Request Changes/Needs Work)
    - Key highlights and concerns
    
    ## 🔒 Critical Issues ({len(critical_issues)} found)
    - Security vulnerabilities
    - Bugs that must be fixed before merge
    
    ## 💡 Suggestions for Improvement ({len(improvements)} found)
    - Code quality improvements
    - Performance optimizations
    - Best practice recommendations
    
    ## ✅ Positive Aspects ({len(positive_feedback)} found)
    - Well-implemented features
    - Good practices observed
    
    ## 🚀 Recommendation
    - Clear next steps
    - Priority of issues to address
    
    Detailed Review Comments:
    {all_comments}
    
    Provide a clear, actionable summary that helps the developer improve their code.
    """
    
    try:
        messages = [
            SystemMessage(content="You are a senior technical lead providing a comprehensive, professional code review summary."),
            HumanMessage(content=summary_prompt)
        ]
        
        response = llm.invoke(messages)
        
        return {
            **state,
            "final_summary": response.content
        }
    except Exception as e:
        return {
            **state,
            "error": f"Error generating summary: {str(e)}"
        }

def post_github_comments(state: PRReviewState) -> PRReviewState:
    """
    Node to post review comments directly to GitHub PR
    """
    print("💬 Posting review comments to GitHub...")
    
    try:
        repo = github_client.get_repo(f"{state['repo_owner']}/{state['repo_name']}")
        pr = repo.get_pull(state['pr_number'])
        
        # Post summary comment
        if state.get("final_summary"):
            summary_comment = f"""# 🤖 AI Code Review Summary

{state['final_summary']}

---
*This review was automatically generated by AI PR Review Agent*
*Review completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*
"""
            pr.create_issue_comment(summary_comment)
            print("✅ Posted summary comment")
        
        # Post individual file review comments
        comments_posted = 0
        for review_comment in state.get("review_comments", []):
            if review_comment["type"] in ["ai_review", "info"] and review_comment.get("github_comment"):
                try:
                    pr.create_issue_comment(review_comment["github_comment"])
                    comments_posted += 1
                    print(f"✅ Posted comment for {review_comment['file']}")
                except Exception as e:
                    print(f"⚠️ Failed to post comment for {review_comment['file']}: {str(e)}")
        
        print(f"✅ Successfully posted {comments_posted + 1} comments to GitHub")
        
        return {
            **state,
            "github_comments_posted": True
        }
        
    except Exception as e:
        error_msg = f"Error posting GitHub comments: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            **state,
            "error": error_msg,
            "github_comments_posted": False
        }

def should_continue_after_fetch(state: PRReviewState) -> str:
    """
    Conditional edge function after fetching PR data
    """
    if state.get("error"):
        return "error_handler"
    elif not state.get("files_changed"):
        return "no_files_handler"
    elif len(state.get("files_changed", [])) > 50:
        return "large_pr_handler"
    else:
        return "analyze_code_changes"

def should_continue_after_analysis(state: PRReviewState) -> str:
    """
    Conditional edge function after code analysis
    """
    if state.get("error"):
        return "error_handler"
    elif not state.get("review_comments"):
        return "no_comments_handler"
    else:
        return "generate_summary"

def should_continue_after_summary(state: PRReviewState) -> str:
    """
    Conditional edge function after summary generation
    """
    if state.get("error"):
        return "error_handler"
    else:
        return "post_github_comments"

def should_continue_after_posting(state: PRReviewState) -> str:
    """
    Conditional edge function after posting comments
    """
    if state.get("error"):
        return "error_handler"
    else:
        return END

def error_handler(state: PRReviewState) -> PRReviewState:
    """
    Handle errors in the workflow
    """
    print(f"❌ Error occurred: {state.get('error', 'Unknown error')}")
    return state

def no_files_handler(state: PRReviewState) -> PRReviewState:
    """
    Handle case when no files are changed
    """
    print("⚠️  No files changed in this PR")
    return {
        **state,
        "final_summary": "This PR contains no file changes to review. This might be a documentation-only update or the PR might have issues.",
        "github_comments_posted": False
    }

def no_comments_handler(state: PRReviewState) -> PRReviewState:
    """
    Handle case when no review comments are generated
    """
    print("⚠️  No review comments generated")
    return {
        **state,
        "final_summary": "No specific issues found during automated review. The changes appear to follow good practices and are ready for manual review.",
        "github_comments_posted": False
    }

def large_pr_handler(state: PRReviewState) -> PRReviewState:
    """
    Handle large PRs that might hit API limits
    """
    file_count = len(state.get("files_changed", []))
    print(f"⚠️  Large PR detected ({file_count} files). Providing high-level analysis...")
    
    # Analyze only the most important files
    important_files = []
    for file_data in state["files_changed"][:20]:  # Limit to first 20 files
        if any(ext in file_data["filename"] for ext in ['.py', '.js', '.ts', '.tsx', '.java', '.go', '.cpp', '.c']):
            important_files.append(file_data)
    
    return {
        **state,
        "files_changed": important_files
    }

# Build the Graph
def create_pr_review_graph():
    """
    Create and configure the LangGraph workflow with conditional logic
    """
    # Initialize the graph
    workflow = StateGraph(PRReviewState)
    
    # Add nodes
    workflow.add_node("fetch_pr_data", fetch_pr_data)
    workflow.add_node("analyze_code_changes", analyze_code_changes)
    workflow.add_node("generate_summary", generate_summary)
    workflow.add_node("post_github_comments", post_github_comments)
    workflow.add_node("error_handler", error_handler)
    workflow.add_node("no_files_handler", no_files_handler)
    workflow.add_node("no_comments_handler", no_comments_handler)
    workflow.add_node("large_pr_handler", large_pr_handler)
    
    # Define the entry point
    workflow.set_entry_point("fetch_pr_data")
    
    # Add conditional edges using should_continue functions
    workflow.add_conditional_edges(
        "fetch_pr_data",
        should_continue_after_fetch,
        {
            "analyze_code_changes": "analyze_code_changes",
            "error_handler": "error_handler",
            "no_files_handler": "no_files_handler",
            "large_pr_handler": "large_pr_handler"
        }
    )
    
    # Large PR handler goes to analysis with limited files
    workflow.add_edge("large_pr_handler", "analyze_code_changes")
    
    workflow.add_conditional_edges(
        "analyze_code_changes",
        should_continue_after_analysis,
        {
            "generate_summary": "generate_summary",
            "error_handler": "error_handler",
            "no_comments_handler": "no_comments_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "generate_summary",
        should_continue_after_summary,
        {
            "post_github_comments": "post_github_comments",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "post_github_comments",
        should_continue_after_posting,
        {
            "error_handler": "error_handler",
            END: END
        }
    )
    
    # Add edges from handler nodes to END
    workflow.add_edge("error_handler", END)
    workflow.add_edge("no_files_handler", END)
    workflow.add_edge("no_comments_handler", END)
    
    # Compile the graph
    return workflow.compile()

# Main execution function
def review_pr(pr_url: str, detailed_output: bool = False):
    """
    Main function to review a GitHub PR
    """
    print(f"🚀 Starting AI-powered GitHub PR review for: {pr_url}")
    print("="*80)
    
    # Create the graph
    app = create_pr_review_graph()
    
    # Initial state
    initial_state = {
        "pr_url": pr_url,
        "repo_owner": "",
        "repo_name": "",
        "pr_number": 0,
        "pr_data": None,
        "files_changed": [],
        "review_comments": [],
        "final_summary": None,
        "github_comments_posted": False,
        "webhook_event": None,
        "error": None
    }
    
    # Run the graph
    try:
        result = app.invoke(initial_state)
        
        if result.get("error"):
            print(f"❌ Error: {result['error']}")
            return result
        
        # Display results
        print("\n" + "="*80)
        print("📋 AI GITHUB PR REVIEW COMPLETED")
        print("="*80)
        print(result["final_summary"])
        print("="*80)
        
        if result.get("github_comments_posted"):
            print("💬 ✅ Comments posted to GitHub PR")
        else:
            print("💬 ⚠️ Comments not posted to GitHub")
        
        if detailed_output and result.get("review_comments"):
            print("\n🔍 DETAILED FILE-BY-FILE REVIEW:")
            print("="*80)
            for comment in result["review_comments"]:
                if comment["type"] != "error":
                    print(f"\n📁 **{comment['file']}** ({comment.get('file_status', 'unknown')})")
                    print(f"Changes: {comment.get('changes_count', 'unknown')}")
                    print("-" * 60)
                    print(comment['comment'])
                    print("-" * 60)
        
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error during PR review: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}

def review_pr_from_webhook(pr_url: str, webhook_payload: dict):
    """
    Review PR triggered from webhook with additional context
    """
    print(f"🔗 Processing webhook-triggered review for: {pr_url}")
    
    # Create the graph
    app = create_pr_review_graph()
    
    # Initial state with webhook context
    initial_state = {
        "pr_url": pr_url,
        "repo_owner": "",
        "repo_name": "",
        "pr_number": 0,
        "pr_data": None,
        "files_changed": [],
        "review_comments": [],
        "final_summary": None,
        "github_comments_posted": False,
        "webhook_event": webhook_payload,
        "error": None
    }
    
    try:
        result = app.invoke(initial_state)
        print(f"🎉 Webhook review completed for PR: {pr_url}")
        return result
    except Exception as e:
        error_msg = f"Webhook review error: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}

def check_github_access():
    """
    Check if GitHub API access is working
    """
    try:
        user = github_client.get_user()
        print(f"✅ GitHub API access confirmed. Authenticated as: {user.login}")
        rate_limit = github_client.get_rate_limit()
        print(f"📊 API Rate Limit: {rate_limit.core.remaining}/{rate_limit.core.limit}")
        return True
    except Exception as e:
        print(f"❌ GitHub API access failed: {str(e)}")
        return False

def start_webhook_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the webhook server
    """
    print(f"🚀 Starting GitHub PR Review Agent webhook server...")
    print(f"📡 Server will run on http://{host}:{port}")
    print(f"🔗 Webhook endpoint: http://{host}:{port}/webhook/github")
    print(f"❤️ Health check: http://{host}:{port}/health")
    
    if not check_github_access():
        print("❌ GitHub access check failed. Please check your GITHUB_TOKEN.")
        return
    
    uvicorn.run(app, host=host, port=port)

# Example usage and CLI interface
if __name__ == "__main__":
    import sys
    
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY in your .env file")
        exit(1)
    
    if not os.getenv("GITHUB_TOKEN"):
        print("❌ Please set GITHUB_TOKEN in your .env file")
        exit(1)
    
    # Command line argument handling
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "server":
            # Start webhook server
            host = sys.argv[2] if len(sys.argv) > 2 else "0.0.0.0"
            port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
            start_webhook_server(host, port)
            
        elif command.startswith("http"):
            # Direct PR review
            pr_url = command
            detailed = "--detailed" in sys.argv
            result = review_pr(pr_url, detailed_output=detailed)
            
            if not result.get("error"):
                print(f"\n🎉 Review completed successfully!")
                if result.get("pr_data"):
                    pr_data = result["pr_data"]
                    print(f"📈 PR Stats: +{pr_data['additions']}/-{pr_data['deletions']} across {pr_data['changed_files']} files")
            else:
                print(f"\n💥 Review failed: {result['error']}")
                exit(1)
        else:
            print("Usage:")
            print("  python pr_review_agent.py server [host] [port]  # Start webhook server")
            print("  python pr_review_agent.py <PR_URL> [--detailed]  # Review specific PR")
            print("  python pr_review_agent.py  # Interactive mode")
    else:
        # Interactive mode
        print("🤖 GitHub PR Review Agent")
        print("1. Type 'server' to start webhook server")
        print("2. Enter a GitHub PR URL to review")
        
        user_input = input("Choice: ").strip()
        
        if user_input.lower() == "server":
            start_webhook_server()
        elif user_input.startswith("http"):
            result = review_pr(user_input, detailed_output=True)
        else:
            print("❌ Invalid input. Please enter 'server' or a GitHub PR URL.")
            exit(1)