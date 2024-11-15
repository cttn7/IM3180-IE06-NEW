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

// Function to check if user is at the bottom of the page
function checkScrollPosition() {
    const footer = document.querySelector("footer");

    // Check if the user is at the bottom of the page
    const atBottom = (window.innerHeight + window.pageYOffset) >= (document.documentElement.scrollHeight - 1);

    if (atBottom) {
        footer.style.display = "block"; // Show footer
    } else {
        footer.style.display = "none"; // Hide footer
    }
}

// Attach the scroll event listener to window
window.addEventListener("scroll", checkScrollPosition);

// Function to display fetched image data in the #results div
function displayImages(data) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = ''; // Clear previous content

    if (data.length > 0) {
        data.forEach(item => {
            const imageContainer = document.createElement('div');
            imageContainer.className = 'image-card';
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
        resultsDiv.innerHTML = '<p>No posts found.</p>';
    }
}

// Fetch and display data
function fetchAndDisplay(url) {
    fetch(url)
        .then(response => response.json())
        .then(data => {
            displayImages(data);
        })
        .catch(err => {
            console.error('Error fetching data:', err);
            document.getElementById('results').innerHTML = '<p>Error loading data</p>';
        });
}

// Function to perform search with category filter
function performSearch() {
    const searchTerm = document.getElementById('search-input').value.trim();
    const category = document.getElementById('category-filter').value;

    // Show loading screen during the fetch
    document.getElementById('loadingScreen').style.display = 'block';

    // Construct API URL based on search term and category
    let apiUrl = 'http://localhost:3000/posts';
    
    if (searchTerm) {
        apiUrl = `http://localhost:3000/search?query=${encodeURIComponent(searchTerm)}`;
    }

    if (category) {
        apiUrl += (searchTerm ? `&category=${encodeURIComponent(category)}` : `?category=${encodeURIComponent(category)}`);
    }

    console.log("API URL:", apiUrl); // Debugging line

    // Fetch the results and display them
    fetch(apiUrl)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            document.getElementById('loadingScreen').style.display = 'none'; // Hide loading screen
            displayImages(data); // Display fetched data
        })
        .catch(error => {
            document.getElementById('loadingScreen').style.display = 'none'; // Hide loading screen on error
            console.error('Error fetching search results:', error);
            document.getElementById('results').innerHTML = '<p>Error fetching results. Please try again later.</p>';
        });
}

// Fetch all posts on page load
document.addEventListener('DOMContentLoaded', () => {
    fetchAndDisplay('http://localhost:3000/posts');
});

// Trigger search when Enter key is pressed
document.getElementById('search-input').addEventListener('keyup', function(event) {
    if (event.key === 'Enter') {
        performSearch();
    }
});

// Handle search button click
document.getElementById('search-button').onclick = function() {
    performSearch();
};

// Handle category filter change to auto-fetch data
document.getElementById('category-filter').addEventListener('change', function() {
    performSearch(); // Automatically fetch results based on selected category
});

// mute button speaker on off function
document.addEventListener('DOMContentLoaded', function() {
    const audio = document.getElementById('backgroundMusic');
    const muteButton = document.getElementById('muteButton');
    const speakerIcon = document.getElementById('speakerIcon');

    // Play the background music when the page loads
    audio.play();

    // Set up event listener for the mute button
    muteButton.addEventListener('click', function() {
        // Toggle mute
        if (audio.muted) {
            audio.muted = false;
            // Change the icon to the unmuted speaker image
            speakerIcon.src = 'assets/images/speaker-on.png';  // Change to your unmuted icon
        } else {
            audio.muted = true;
            // Change the icon to the muted speaker image
            speakerIcon.src = 'assets/images/speaker-off.png'; // Change to your muted icon
        }
    });
});



//Add click event-listener to search button and function to execute when clicked
/*
document.getElementById('search-button').addEventListener('click', () => {
    const query = document.getElementById('search-input').value;

    // Call function to send the query to the backend
    fetchAndDisplay(`http://localhost:3000/search?query=${encodeURIComponent(query)}`)
});
*/

/*document.getElementById('search-button').addEventListener('click', () => {
    const query = document.getElementById('search-input').value;

    // Send the query to the backend (Line 5 to Line 51)
    fetch(`http://localhost:3000/search?query=${encodeURIComponent(query)}`)
        .then(response => response.json())
        .then(data => {
            // Display the results
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '';

            if (data.length > 0) {
                data.forEach(item => {
                    resultsDiv.innerHTML += `
                        <div class="image-card">
                            <h3>${item.title}</h3>
                            <img src="${item.image_path}" alt="${item.title}" width="280">
                            <p>${item.description}</p>
                            <a href="${item.link}" target="_blank">More Info</a>
                        </div>
                    `;
                });
            } else {
                resultsDiv.innerHTML = '<p>No results found</p>';
            }
        })
        .catch(err => {
            console.error('Error:', err);
        });
});
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
*/