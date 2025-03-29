document.addEventListener("DOMContentLoaded", function () {
    let profiles = document.querySelectorAll(".profile");
    let currentIndex = 0;

    function showProfile(index) {
        profiles.forEach((profile, i) => {
            profile.classList.remove("active");
            if (i === index) {
                profile.classList.add("active");
            }
        });
    }

    document.querySelector(".prev").addEventListener("click", function () {
        currentIndex = (currentIndex - 1 + profiles.length) % profiles.length;
        showProfile(currentIndex);
    });

    document.querySelector(".next").addEventListener("click", function () {
        currentIndex = (currentIndex + 1) % profiles.length;
        showProfile(currentIndex);
    });

    // Show the first profile initially
    showProfile(currentIndex);
});
