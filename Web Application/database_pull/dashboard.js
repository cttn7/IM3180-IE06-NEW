document.addEventListener("DOMContentLoaded", () => {
    const postId = localStorage.getItem('selectedPostId');
    
    // Function to display a post in the main content area
    const displayPostDetails = (post) => {
        const postDetailsContainer = document.getElementById('post-details');
        postDetailsContainer.innerHTML = `
            <h2>${post.title}</h2>
            <img src="${post.image_path}" alt="${post.title}" style="max-width: 100%; height: auto;">
            <p>${post.description}</p>
            <p>${post.summary}</p>
        `;
    };

    // Fetch the list of posts for the sidebar
    fetch('http://localhost:3000/posts')
        .then(response => response.json())
        .then(posts => {
            const postListContainer = document.getElementById('post-list');
            postListContainer.innerHTML = ''; // Clear existing content

            // Display each post as a link in the sidebar
            posts.forEach(post => {
                const postItem = document.createElement('div');
                postItem.classList.add('post-item');
                postItem.innerHTML = `
                    <h3>${post.title}</h3>
                `;
                postItem.onclick = () => displayPostDetails(post); // Display post on click
                postListContainer.appendChild(postItem);
            });

            // If a specific post ID is selected, load its details
            if (postId) {
                const selectedPost = posts.find(post => post.id == postId);
                if (selectedPost) {
                    displayPostDetails(selectedPost);
                }
                // Clear postId from localStorage if desired
                localStorage.removeItem('selectedPostId');
            } else if (posts.length > 0) {
                // Default to first post if no specific ID is selected
                displayPostDetails(posts[0]);
            }
        })
        .catch(error => {
            console.error('Error fetching posts:', error);
            document.getElementById('post-list').innerHTML = '<p>Error loading posts. Please try again later.</p>';
        });
});
