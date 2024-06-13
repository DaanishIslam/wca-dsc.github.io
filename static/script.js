document.addEventListener('DOMContentLoaded', function () {
    const uploadForm = document.getElementById('upload-form');
    const executeButton = document.getElementById('execute-button');
    const messageContainer = document.getElementById('message-container');
    const loader = document.querySelector('.loader');

    // Get and save the current domain name
    const currentDomain = window.location.hostname;

    // Clear form inputs on page load
    uploadForm.reset();

    // Handle file upload
    uploadForm.addEventListener('submit', function (event) {
        event.preventDefault();
        const formData = new FormData(uploadForm);

        fetch('/upload', {
            method: 'POST',
            body: formData
        }).then(response => response.json())
        .then(data => {
            messageContainer.innerHTML = ''; // Clear previous messages
            const message = document.createElement('p');
            message.textContent = data.message;
            messageContainer.appendChild(message);

            if (data.success) {
                executeButton.disabled = false; // Enable the execute button
            } else {
                executeButton.disabled = true; // Disable the execute button
            }
        }).catch(error => {
            console.error('Error:', error);
        });
    });

    // Handle script execution
    executeButton.addEventListener('click', function () {
        loader.style.display = 'block'; // Show the loader
        fetch('/execute', {
            method: 'POST'
        }).then(response => response.json())
        .then(data => {
            loader.style.display = 'none'; // Hide the loader
            messageContainer.innerHTML = `<p>${data.message}</p>`;

            if (data.success) {
                setTimeout(function () {
                    window.location.href = '/plots';
                }, 1000);
            }
        }).catch(error => {
            loader.style.display = 'none'; // Hide the loader in case of error
            console.error('Error:', error);
        });
    });

    // Handle link clicks for AJAX navigation
    document.addEventListener('click', function (event) {
        if (event.target.matches('a')) {
            event.preventDefault();
            loader.style.display = 'block'; // Show the loader
            
            // Fetch the href and navigate after loading is complete
            const href = event.target.getAttribute('href');
            fetch(href, {
                method: 'GET'
            }).then(response => {
                if (response.ok) {
                    window.location.href = href; // Navigate to the link after successful fetch
                } else {
                    loader.style.display = 'none'; // Hide the loader if fetch fails
                    console.error('Failed to fetch the link');
                }
            }).catch(error => {
                loader.style.display = 'none'; // Hide the loader on error
                console.error('Error:', error);
            });
        }
    });

    // Optionally handle user leaving the domain
    // window.addEventListener('beforeunload', function (event) {
    //     if (window.location.hostname !== currentDomain) {
    //         fetch('/cleanup', {
    //             method: 'POST'
    //         }).then(response => response.json())
    //         .then(data => {
    //             console.log('Cleanup success:', data.message);
    //         }).catch(error => {
    //             console.error('Error during cleanup:', error);
    //         });
    //     }
    // });
});
