
# ReviewScorer

With more and more code being generated using AI, the PR review process is becoming more important.
There is already a lot of tools that measure the quality and perforance of code but there is not
a lot of tools that look at the performance of a review.

This project aims at providing a score for reviews done in a PR to gain insight on how well a
review was performed.

This project mainly uses LLMs to categorise and get a scoring for a review done on a PR.
The scores are saved to a local RAG, which can be corrected should the user feel the LLM rating
is incorrect. This local RAG will then be used on the next run of this application and the historic
scores of a review will be taken into account when making new reviews.

Main repo for ReviewScorer, currently just a POC.

## How it works

This is a command-line tool that accepts a PR url as its input. A scoring of the review will be done
against the comments of the PR url. The application will only score comments that it determines to
have resulted in a code change in the PR or have been helpful to the PR creator in terms of learnings
and tips. The application will also ignore comments it determines was made by an AI.

On startup, the application will make a connection to the AI model and Github MCP server using
information defined in the .env file. It will then load up the RAG of historic reviews that have already
been done.

Then the review scoring process is run in 2 sequences

-Getting the relevant comments
    -Get the code changes that were made at the time the PR was created using AI tools and Github MCP
    -Let the LLM perform a PR review on the code changes gotten
    -Get all the comments made on the PR and exclude the ones that were made by the LLM in the last step
    -From the leftover comments, select the comments that resulted in code change, learnings and tips

-Scoring the comments
    -For each comment that needs scoring, fetch 10 closest matching historic reviews from RAG
    -The LLM performs a review on the comment while considering the historical reviews
    -The LLM outputs the score of a comment in a structured output
    -The score is then added to the RAG and saved to a file

## Contributing

Any contributions are welcome.

## License

This project is licensed under the MIT License.

## Links

* Github repo: https://github.com/jstadevcoder/ReviewScorer
