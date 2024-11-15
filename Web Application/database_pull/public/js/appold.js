// Function to display fetched image data in the #results div
function displayImages(data) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = ''; // Clear previous content in white box

    if (data.length > 0) {
        data.forEach(item => {
            // Create HTML structure for each image item
            const imageContainer = document.createElement('div');
            imageContainer.className = 'image-card'; //assign a className for css styles
            imageContainer.innerHTML = `
                <a href="dashboard.html" style="text-decoration: none; color: inherit;" onclick="localStorage.setItem('selectedPostId', ${item.id})">
                    <h3>${item.title}</h3>
                    <img src="${item.image_path}" alt="${item.title}" style="max-width: 100%; height: auto;">
                    
                    <p>${item.summary}</p>
                </a>
            `;
            resultsDiv.appendChild(imageContainer);
        });
    } else {
        resultsDiv.innerHTML = '<p>No images found</p>';
    }
}
// Function to handle fetch request and display data
function fetchAndDisplay(url) {
    fetch(url)
        .then(response => response.json())
        .then(data => {
            displayImages(data); // Use shared function to display images
        })
        .catch(err => {
            console.error('Error fetching data:', err);
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<p>Error loading data</p>';
        });
}
// Fetch all images from index.js. Use DOMContentLoaded EventListener to ensure the DOM //is fully loaded before executing the fetch calls
document.addEventListener('DOMContentLoaded', () => {
    // Call fetchAndDisplay function to Fetch from the /images endpoint
    fetchAndDisplay('http://localhost:3000/posts'); 
});

//Add click event-listener to search button and function to execute when clicked
/*
document.getElementById('search-button').addEventListener('click', () => {
    const query = document.getElementById('search-input').value;

    // Call function to send the query to the backend
    fetchAndDisplay(http://localhost:3000/search?query=${encodeURIComponent(query)})
});
*/
// Example function to fetch data and update the chart
function fetchData() {
    const query = document.getElementById('search').value;
    
    if (!query) {
        alert('Please enter a search query.');
        return;
    }

    // Simulate a data fetch with mock data
    const mockData = {
        labels: ['Positive', 'Negative', 'Neutral'],
        datasets: [{
            label: 'Sentiment Analysis',
            data: [40, 35, 25],  // Replace this with real data
            backgroundColor: ['#28a745', '#dc3545', '#ffc107'],
        }]
    };

    renderChart(mockData);
}

// Function to render chart with data
function renderChart(data) {
    const ctx = document.getElementById('chartContainer').getContext('2d');

    new Chart(ctx, {
        type: 'doughnut',
        data: data,
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
            },
        }
    });
}
