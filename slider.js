// slider.js

// Grab all slider containers
const sliderContainers = document.querySelectorAll(".slider-container");

// We’ll keep an array of slider states
const sliders = [];

// For each container, gather slides, track the current index, set up an interval
sliderContainers.forEach(container => {
  const slides = container.querySelectorAll(".slide");
  let currentSlide = 0;

  // Show the first slide
  slides[currentSlide].classList.add("active");

  // Every 3 seconds, fade to the next
  setInterval(() => {
    slides[currentSlide].classList.remove("active");
    currentSlide = (currentSlide + 1) % slides.length;
    slides[currentSlide].classList.add("active");
  }, 7000); // 7 seconds
});
