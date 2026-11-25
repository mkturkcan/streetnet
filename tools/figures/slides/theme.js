// Theme of a figure slide from ?theme=light|dark (tools/figures/render.py renders both).
// Images with data-src="name-{theme}.png" load the variant of the current theme.
(function () {
  var theme = new URLSearchParams(location.search).get("theme") || "light";
  document.documentElement.setAttribute("data-theme", theme);
  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("img[data-src]").forEach(function (im) {
      im.src = im.getAttribute("data-src").replace("{theme}", theme);
    });
  });
})();
