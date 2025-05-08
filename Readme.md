The API for generating a set of multiple-choice questions starting from a context. The model will return a list of questions and 4 answer options for each one, with an explanation for each option.

# API usage

You can use our Flask server to send requests for scoring. The request must come in the form of a POST to https://chat.readerbench.com/quiz/generate.
An API key is required to access the deployed service. The key should be added in an .env file using the name "API_TOKEN".

curl --location 'https://chat.readerbench.com/quiz/generate' \
--header 'Content-Type: application/json' \
--data '{
    "context": "Lincoln was born into poverty in Kentucky and raised on the frontier. He was self-educated and became a lawyer, Illinois state legislator, and U.S. representative. Angered by the Kansas–Nebraska Act of 1854, which opened the territories to slavery, he became a leader of the new Republican Party. He reached a national audience in the 1858 Senate campaign debates against Stephen A. Douglas. Lincoln won the 1860 presidential election, but the South viewed his election as a threat to slavery, and Southern states began seceding to form the Confederate States of America. A month after Lincoln assumed the presidency, Confederate forces attacked Fort Sumter, starting the Civil War.",
    "num_questions": 5,
    "token": "API_TOKEN"
}'
