%%writefile static/script.js
document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const uploadSection = document.getElementById('upload-section');
    const chatSection = document.getElementById('chat-section');
    const pdfFileInput = document.getElementById('pdf-file');
    const fileLabel = document.querySelector('.file-label');
    const fileNameSpan = document.getElementById('file-name');
    const uploadButton = document.getElementById('upload-button');
    const statusMessage = document.getElementById('status-message');
    const chatWindow = document.getElementById('chat-window');
    const questionInput = document.getElementById('question-input');
    const sendButton = document.getElementById('send-button');

    let selectedFile = null;

    // --- File Upload Logic ---
    fileLabel.addEventListener('click', () => pdfFileInput.click());
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        fileLabel.addEventListener(eventName, preventDefaults, false);
    });
    ['dragenter', 'dragover'].forEach(eventName => {
        fileLabel.addEventListener(eventName, () => fileLabel.classList.add('dragover'), false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
        fileLabel.addEventListener(eventName, () => fileLabel.classList.remove('dragover'), false);
    });

    fileLabel.addEventListener('drop', handleDrop, false);
    pdfFileInput.addEventListener('change', (e) => handleFileSelect(e.target.files));

    function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }
    function handleDrop(e) { handleFileSelect(e.dataTransfer.files); }

    function handleFileSelect(files) {
        if (files.length > 0) {
            selectedFile = files[0];
            if (selectedFile.type === 'application/pdf') {
                fileNameSpan.textContent = selectedFile.name;
            } else {
                selectedFile = null;
                fileNameSpan.textContent = 'Please select a valid PDF file.';
            }
        }
    }

    uploadButton.addEventListener('click', async () => {
        if (!selectedFile) {
            updateStatus('Please select a PDF file first.', 'error');
            return;
        }

        updateStatus('Processing document...', 'loading');
        uploadButton.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/upload-pdf', { method: 'POST', body: formData });
            const result = await response.json();
            if (response.ok && result.status === 'success') {
                updateStatus(result.message, 'success');
                uploadSection.classList.add('hidden');
                chatSection.classList.remove('hidden');
                addMessageToChat('System', 'The document has been processed. You can now ask questions.');
            } else {
                throw new Error(result.message || 'Failed to process PDF.');
            }
        } catch (error) {
            updateStatus(`Error: ${error.message}`, 'error');
        } finally {
            uploadButton.disabled = false;
        }
    });

    // --- Chat Logic ---
    sendButton.addEventListener('click', sendQuestion);
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuestion();
        }
    });

    async function sendQuestion() {
        const question = questionInput.value.trim();
        if (!question) return;

        addMessageToChat('You', question);
        questionInput.value = '';
        questionInput.style.height = 'auto';
        const botMessageElement = addMessageToChat('Bot', '', true);

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question }),
            });
            const result = await response.json();
            botMessageElement.classList.remove('typing');
            if (response.ok) {
                botMessageElement.innerHTML = result.answer.replace(/\n/g, '<br>');
            } else {
                throw new Error(result.answer || 'An error occurred.');
            }
        } catch (error) {
            botMessageElement.textContent = `Error: ${error.message}`;
        }
    }

    function addMessageToChat(sender, text, isTyping = false) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', sender.toLowerCase() + '-message');
        if (isTyping) { messageElement.classList.add('typing'); }
        messageElement.innerHTML = text.replace(/\n/g, '<br>');
        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        return messageElement;
    }

    function updateStatus(message, type) {
        // CHANGED: Use a CSS loader instead of an emoji
        const loaderHTML = type === 'loading' ? '<div class="loader"></div>' : '';
        statusMessage.innerHTML = `${loaderHTML}<span>${message}</span>`;
        statusMessage.className = `status ${type}`;
    }

    // Auto-resize textarea
    questionInput.addEventListener('input', () => {
        questionInput.style.height = 'auto';
        questionInput.style.height = (questionInput.scrollHeight) + 'px';
    });
});