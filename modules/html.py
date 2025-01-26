progress_html = '''
<div class="loader-container">
  <div class="loader"></div>
  <div class="progress-container">
    <progress value="*number*" max="100"></progress>
  </div>
  <span>*text*</span>
</div>
'''

css = '''
.loader-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
}

.loader {
    border: 8px solid #f3f3f3;
    border-top: 8px solid #3498db;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    animation: spin 1s linear infinite;
}

.progress-container {
    width: 100%;
    max-width: 300px;
}

progress {
    width: 100%;
    height: 20px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
'''


def make_progress_html(number, text):
    return progress_html.replace('*number*', str(number)).replace('*text*', text)
