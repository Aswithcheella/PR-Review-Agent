import os
import re
from dotenv import load_dotenv
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from github import Github, GithubException
import base64

load_dotenv()

#Stae Schema - this defines the structure of the data that flows through the state graph
class PRReviewState(TypedDict):
    pr_url: str
    repo_owner: str
    repo_name: str
    pr_number: int
    pr_data: Optional[dict]
    files_changed: List[str]
    review_comments: List[str]
    final_summary: Optional[str]
    error: Optional[str]

#initialize the llm
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)

github_client = Github(os.getenv("GITHUB_TOKEN"))

def parse_pr_url(pr_url: str) -> tuple:
    """
    Parse GitHub PR URL to extract owner, repo, and PR number
    """
    pattern = r'https://github\.com/([^/]+)/([^/]+)/pull/(\d+)'
    match = re.match(pattern, pr_url)
    
    if not match:
        raise ValueError(f"Invalid GitHub PR URL format: {pr_url}")
    
    return match.group(1), match.group(2), int(match.group(3))

# node to fetch PR data
def fetch_pr_data(State: PRReviewState) -> PRReviewState:
    """Need to fetch PR data from the Github API.
       for now we will use a boilerplate data."""
    # Simulating fetching PR data
    print("🔍 Fetching PR data...")

    try:
        repo_owner, repo_name, pr_number = parse_pr_url(State["pr_url"])
        repo = github_client.get_repo(f"{repo_owner}/{repo_name}")
        pr = repo.get_pull(pr_number)

        # fetch PR files
        files = list(pr.get_files())

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
        
        #get PR metadata
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
            **State,
            "repo_owner": repo_owner,
            "repo_name": repo_name,
            "pr_number": pr_number,
            "pr_data": pr_data,
            "files_changed": files_changed
        }
    except GithubException as e:
        error_msg = f"GitHub API error: {e.data.get('message', str(e)) if hasattr(e, 'data') else str(e)}"
        print(f"❌ {error_msg}")
        return {**State, "error": error_msg}
    except ValueError as e:
        error_msg = f"URL parsing error: {str(e)}"
        print(f"❌ {error_msg}")
        return {**State, "error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error fetching PR data: {str(e)}"
        print(f"❌ {error_msg}")
        return {**State, "error": error_msg}

# node to analyze code changes
def analyze_code_changes(State: PRReviewState) -> PRReviewState:
    """
    Node to analyze the code changes in the PR using AI
    """
    print("🔍 Analyzing code changes with AI...")
    
    if not State.get("files_changed"):
        return {**State, "error": "No files to analyze"}
    
    review_comments = []
    pr_context = State["pr_data"]
    
    # Create context about the PR
    pr_context_text = f"""
    PR Title: {pr_context['title']}
    Description: {pr_context['description'][:500]}...
    Author: {pr_context['author']}
    Base Branch: {pr_context['base_branch']} → Head Branch: {pr_context['head_branch']}
    Total Changes: +{pr_context['additions']}/-{pr_context['deletions']} across {pr_context['changed_files']} files
    """
    
    for file_change in State["files_changed"]:
        print(f"  📄 Analyzing {file_change['filename']}...")
        
        # Skip binary files or very large files
        if not file_change.get("patch") and file_change["changes"] > 100:
            review_comments.append({
                "file": file_change['filename'],
                "comment": "⚠️ Large file or binary file - manual review recommended",
                "type": "info",
                "line_number": None
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
        Format your response with clear sections and actionable suggestions."""
        
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
        
        Please provide a detailed code review for this file.
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = llm.invoke(messages)
            review_comments.append({
                "file": file_change['filename'],
                "comment": response.content,
                "type": "ai_review",
                "line_number": None,
                "file_status": file_change['status'],
                "changes_count": file_change['changes']
            })
            
        except Exception as e:
            review_comments.append({
                "file": file_change['filename'],
                "comment": f"❌ Error analyzing file: {str(e)}",
                "type": "error",
                "line_number": None
            })
    
    print(f"✅ Analyzed {len(review_comments)} files")
    return {
        **State,
        "review_comments": review_comments
    }

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

# node to generate review summary or comments
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
        "final_summary": "This PR contains no file changes to review. This might be a documentation-only update or the PR might have issues."
    }

def no_comments_handler(state: PRReviewState) -> PRReviewState:
    """
    Handle case when no review comments are generated
    """
    print("⚠️  No review comments generated")
    return {
        **state,
        "final_summary": "No specific issues found during automated review. The changes appear to follow good practices and are ready for manual review."
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
        "files_changed": important_files,
        "final_summary": f"⚠️ This is a large PR with {file_count} files. Automated review focused on the first 20 most critical files. Manual review recommended for completeness."
    }

# build the graph
def create_pr_review_graph():
    """Create and configure the LangGraph workflow for PR review."""
    # Initialize the state with default values
    workflow = StateGraph(PRReviewState)
    # define node in the workflow
    workflow.add_node("fetch_pr_data", fetch_pr_data)
    workflow.add_node("analyze_code_changes", analyze_code_changes)
    workflow.add_node("generate_summary", generate_summary)
    workflow.add_node("error_handler", error_handler)
    workflow.add_node("no_files_handler", no_files_handler)
    workflow.add_node("no_comments_handler", no_comments_handler)
    workflow.add_node("large_pr_handler", large_pr_handler)

    # Set the entry point of the workflow
    workflow.add_edge(START, "fetch_pr_data")

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
            END: END,
            "error_handler": "error_handler"
        }
    )


    workflow.add_edge("error_handler", END)
    workflow.add_edge("no_files_handler", END)
    workflow.add_edge("no_comments_handler", END)

    # Compile the workflow
    return workflow.compile()

# Main execution function
def review_pr(pr_url: str, detailed_output: bool = False):
    """Review a pull request by its URL."""
    print(f"🚀 Starting AI-powered GitHub PR review for: {pr_url}")
    print("="*80)

    # create the graph
    app = create_pr_review_graph()

    # Initialize the state with the PR URL
    initial_state = {
        "pr_url": pr_url,
        "repo_owner": "",
        "repo_name": "",
        "pr_number": 0,
        "pr_data": None,
        "files_changed": [],
        "review_comments": [],
        "final_summary": None,
        "error": None
    }

    try:
        result = app.invoke(initial_state)
        
        if result.get("error"):
            print(f"❌ Error: {result['error']}")
            return result
        
        # Display results
        print("\n" + "="*80)
        print("📋 AI GITHUB PR REVIEW SUMMARY")
        print("="*80)
        print(result["final_summary"])
        print("="*80)
        
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
    
# Example usage
if __name__ == "__main__":
    # Make sure to set your OPENAI_API_KEY in a .env file
    import sys
    
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY in your .env file")
        exit(1)
    
    if not os.getenv("GITHUB_TOKEN"):
        print("❌ Please set GITHUB_TOKEN in your .env file")
        exit(1)
    
    # Check GitHub access
    if not check_github_access():
        exit(1)
    
    # Get PR URL from command line or use example
    if len(sys.argv) > 1:
        pr_url = sys.argv[1]
        detailed = "--detailed" in sys.argv
    else:
        # Example PR URL - replace with a real PR
        pr_url = input("Enter GitHub PR URL: ").strip()
        if not pr_url:
            print("❌ No PR URL provided")
            exit(1)
        detailed = True
    
    # Run the PR review
    print(f"🤖 Starting AI review of: {pr_url}")
    result = review_pr(pr_url, detailed_output=detailed)
    
    if not result.get("error"):
        print(f"\n🎉 Review completed successfully!")
        if result.get("pr_data"):
            pr_data = result["pr_data"]
            print(f"📈 PR Stats: +{pr_data['additions']}/-{pr_data['deletions']} across {pr_data['changed_files']} files")
    else:
        print(f"\n💥 Review failed: {result['error']}")
        exit(1)



    