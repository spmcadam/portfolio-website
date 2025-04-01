document.addEventListener('DOMContentLoaded', function () {
    const topMenu = document.querySelector('.top-menu');
    if (!topMenu) return;
  
    let lastScrollY = window.scrollY;
    const maxScroll = 300;
  
    window.addEventListener('scroll', () => {
      const currentScrollY = window.scrollY;
      const scrollDiff = currentScrollY - lastScrollY;
  
      if (scrollDiff > 5) {
        topMenu.style.transform = 'translateY(-100%)';
        topMenu.style.opacity = 1 - Math.min(currentScrollY / maxScroll, 1.0);
      } else if (scrollDiff < -5) {
        topMenu.style.transform = 'translateY(0)';
        topMenu.style.opacity = 1;
      }
  
      lastScrollY = currentScrollY;
    });
  });


document.addEventListener('DOMContentLoaded', function () {
    const banner = document.querySelector('.banner');
    if (banner && banner.dataset.bg) {
      banner.style.backgroundImage = `url(${banner.dataset.bg})`;
    }
  });