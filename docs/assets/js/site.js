// StreetNet project page: theme toggle, theme-dependent figures, copy buttons, menu and active section.
(function () {
  var root = document.documentElement;
  var media = window.matchMedia("(prefers-color-scheme: dark)");

  function stored() { try { return localStorage.getItem("theme"); } catch (e) { return null; } }
  function store(v) { try { localStorage.setItem("theme", v); } catch (e) { /* private mode */ } }
  function effective() { return root.getAttribute("data-theme") || (media.matches ? "dark" : "light"); }

  // <picture> sources marked data-dark follow the page theme, also when it differs from the OS setting
  function syncPictures() {
    var dark = effective() === "dark";
    document.querySelectorAll("source[data-dark]").forEach(function (s) { s.media = dark ? "all" : "not all"; });
    var btn = document.getElementById("theme");
    if (btn) btn.setAttribute("aria-label", dark ? "Use the light theme" : "Use the dark theme");
  }
  var t = stored();
  if (t === "light" || t === "dark") root.setAttribute("data-theme", t);

  document.addEventListener("DOMContentLoaded", function () {
    syncPictures();
    media.addEventListener("change", syncPictures);
    var btn = document.getElementById("theme");
    if (btn) btn.addEventListener("click", function () {
      var next = effective() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next); store(next); syncPictures();
    });

    // copy buttons on code blocks
    document.querySelectorAll("pre").forEach(function (pre) {
      var b = document.createElement("button");
      b.className = "copy"; b.type = "button"; b.textContent = "Copy";
      b.addEventListener("click", function () {
        var code = pre.querySelector("code") || pre;
        var text = code.innerText.replace(/\s+$/, "");
        (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject()).then(function () {
          b.textContent = "Copied"; setTimeout(function () { b.textContent = "Copy"; }, 1400);
        }, function () { b.textContent = "Select and copy"; });
      });
      pre.appendChild(b);
    });

    // menu on narrow screens
    var menu = document.getElementById("menu"), sheet = document.getElementById("sheet");
    if (menu && sheet) {
      menu.addEventListener("click", function () {
        var open = sheet.classList.toggle("open"); menu.setAttribute("aria-expanded", open ? "true" : "false");
      });
      sheet.querySelectorAll("a").forEach(function (a) {
        a.addEventListener("click", function () { sheet.classList.remove("open"); menu.setAttribute("aria-expanded", "false"); });
      });
    }

    // highlight the section or heading in view
    var links = Array.prototype.slice.call(document.querySelectorAll(".links a[href^='#'], .toc a[href^='#']"));
    var targets = links.map(function (a) { return document.getElementById(a.getAttribute("href").slice(1)); });
    if ("IntersectionObserver" in window && links.length) {
      var seen = new Map();
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) { seen.set(e.target.id, e.isIntersecting); });
        var first = targets.find(function (el) { return el && seen.get(el.id); });
        links.forEach(function (a) { a.classList.toggle("on", !!first && a.getAttribute("href") === "#" + first.id); });
      }, { rootMargin: "-80px 0px -55% 0px" });
      targets.forEach(function (el) { if (el) io.observe(el); });
    }
  });
})();
