const form = document.getElementById('text-form');
const output = document.getElementById('output');
const suggestions = document.getElementById('suggestions');

// Function to fetch suggestions dynamically
async function getSuggestions(query) {
    if (query.length < 1) {
        suggestions.innerHTML = '';
        return;
    }

    const response = await fetch(`/suggest?q=${query}`);
    const result = await response.json();

    suggestions.innerHTML = '';
    result.forEach(word => {
        const option = document.createElement('option');
        option.value = word;
        suggestions.appendChild(option);
    });
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const prompt = document.getElementById('prompt').value;

    const formData = new FormData(form);
    const response = await fetch('/generate', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    output.innerHTML = `<p>Generated Text:</p><strong>${result.generated_text}</strong>`;
});
