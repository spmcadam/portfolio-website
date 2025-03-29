// slider.js

// Select all slider containers
const sliderContainers = document.querySelectorAll(".slider-container");

sliderContainers.forEach((container) => {
  // All the .slide elements in this container
  const slides = container.querySelectorAll(".slide");
  if (!slides.length) return; // If no slides, skip

  // Optional: find arrows in the same container if they exist
  const leftArrow = container.parentNode.querySelector(".arrow.left");
  const rightArrow = container.parentNode.querySelector(".arrow.right");

  let currentSlide = 0;

  // Show the first slide
  slides[currentSlide].classList.add("active");

  // Automatically cycle every 7 seconds
  let intervalId = setInterval(nextSlide, 7000);

  function nextSlide() {
    slides[currentSlide].classList.remove("active");
    currentSlide = (currentSlide + 1) % slides.length;
    slides[currentSlide].classList.add("active");
  }

  function prevSlide() {
    slides[currentSlide].classList.remove("active");
    currentSlide = (currentSlide - 1 + slides.length) % slides.length;
    slides[currentSlide].classList.add("active");
  }

  // If the container has arrows, hook them up
  if (leftArrow && rightArrow) {
    leftArrow.addEventListener("click", () => {
      prevSlide();
      resetInterval();
    });
    rightArrow.addEventListener("click", () => {
      nextSlide();
      resetInterval();
    });
  }

  // A helper to stop and restart the timer, so user clicks don’t immediately get overwritten
  function resetInterval() {
    clearInterval(intervalId);
    intervalId = setInterval(nextSlide, 7000);
  }
});
