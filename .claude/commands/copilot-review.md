# View GitHub Copilot Review Comments

**Quick command to view all GitHub Copilot inline code review comments for a PR**

## Quick Usage (GraphQL - Recommended)

```bash
# For specific PR number (simpler, more reliable)
gh api graphql -f query='
query {
  repository(owner: "talmolab", name: "sleap-roots") {
    pullRequest(number: PR_NUMBER) {
      reviews(first: 10) {
        nodes {
          author { login }
          comments(first: 50) {
            nodes {
              path
              line
              body
            }
          }
        }
      }
    }
  }
}
' --jq '.data.repository.pullRequest.reviews.nodes[] | select(.author.login | contains("opilot")) | .comments.nodes[] | "File: \(.path):\(.line)\n\(.body)\n" + ("="*80)'
```

## Alternative: REST API

```bash
# View Copilot comments for specific PR (REST API)
gh api repos/{owner}/{repo}/pulls/PR_NUMBER/comments --jq '.[] | "File: \(.path):\(.line // .original_line)\n\(.body)\n" + ("="*80)'

# For current PR dynamically
gh api repos/$(gh repo view --json nameWithOwner -q .nameWithOwner)/pulls/$(gh pr view --json number -q .number)/comments --jq '.[] | "File: \(.path):\(.line // .original_line)\n\(.body)\n" + ("="*80)'
```

## What This Does

1. Fetches all inline code review comments from Copilot
2. Formats each comment showing:
   - File path and line number
   - Comment body
   - Separator line between comments

## Important Notes

- GitHub Copilot inline comments come from user **"Copilot"**
- Review summaries come from **"copilot-pull-request-reviewer[bot]"**
- GraphQL approach can fetch both in one query (more efficient)
- REST API requires separate calls for reviews vs comments

## Get Review Summary

To see the overall review summary from Copilot:

```bash
# GraphQL (gets review body + inline comments in one call)
gh api graphql -f query='
query {
  repository(owner: "talmolab", name: "sleap-roots") {
    pullRequest(number: PR_NUMBER) {
      reviews(first: 10) {
        nodes {
          author { login }
          state
          body
          submittedAt
        }
      }
    }
  }
}
' --jq '.data.repository.pullRequest.reviews.nodes[] | select(.author.login | contains("opilot")) | {state, submitted: .submittedAt, body}'

# REST API
gh api repos/{owner}/{repo}/pulls/PR_NUMBER/reviews --jq '.[] | select(.user.login | contains("copilot")) | {state: .state, submitted_at: .submitted_at, body: .body}'
```

## Typical Copilot Comments Include

- Code quality suggestions
- Error handling improvements
- Type safety recommendations
- Best practice violations
- Inconsistencies with documented patterns
- Missing edge case handling

## Integration with Pre-Merge Checks

This command should be run as part of the pre-merge workflow (Phase 6: Review Feedback) to ensure all Copilot feedback is addressed before merging.

## Example Output

```
File: sleap_roots/trait_pipelines.py:310
Empty string fallback for optional genotype_id creates inconsistent data. Should use `null` or `None` for missing optional values...
================================================================================
File: sleap_roots/lengths.py:56
Missing error handling when computation fails. If the result is None, downstream code has no feedback...
================================================================================
```