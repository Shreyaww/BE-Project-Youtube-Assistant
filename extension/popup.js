document.addEventListener('DOMContentLoaded', function() {
    const btnSummarise = document.getElementById("summarise");
    const btnAsk = document.getElementById("ask");
    const output = document.getElementById("output");
    const questionInput = document.getElementById("question");
    const answerOutput = document.getElementById("answer");

    btnSummarise.addEventListener("click", function() {
        btnSummarise.disabled = true;
        btnSummarise.innerHTML = "Summarising...";

        chrome.tabs.query({ currentWindow: true, active: true }, function(tabs) {
            var url = tabs[0].url;
            var xhr = new XMLHttpRequest();
            xhr.open("GET", "http://127.0.0.1:5000/summary?url=" + url, true);
            xhr.onload = function() {
                var text = xhr.responseText;
                output.innerHTML = text;
                btnSummarise.disabled = false;
                btnSummarise.innerHTML = "Summarise";
            }
            xhr.send();
        });
    });

    btnAsk.addEventListener("click", function() {
        var question = questionInput.value.trim();

        if (question !== "") {
            answerOutput.innerText = "Loading answer...";

            var xhr = new XMLHttpRequest();
            xhr.open("POST", "http://127.0.0.1:5000/chat", true);
            xhr.setRequestHeader("Content-Type", "application/json");
            xhr.onreadystatechange = function() {
                if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
                    var response = JSON.parse(xhr.responseText);
                    answerOutput.innerText = response.chatbot_response;
                } else {
                    answerOutput.innerText = "Error fetching answer.";
                }
            };
            xhr.send(JSON.stringify({ question: question }));
        } else {
            answerOutput.innerText = "Please enter a question.";
        }
    });
});
