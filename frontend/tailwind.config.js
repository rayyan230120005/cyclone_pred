/** @type {import('tailwindcss').Config} */
export default {
  // Tailwind scans these files for class names and emits only what it finds.
  // index.html is included because <body> carries utility classes.
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        // Matches the Google Fonts loaded in index.html
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
        hud: ["Orbitron", "monospace"],
      },
      colors: {
        // Mirrors the CSS custom properties in src/index.css so utilities and
        // hand-written rules stay on one palette.
        hud: {
          bg: "#070b14",
          panel: "#0d1527",
          cyan: "#38bdf8",
          rose: "#f43f5e",
          amber: "#fbbf24",
          emerald: "#10b981",
          purple: "#c084fc",
        },
      },
      backdropBlur: {
        xs: "2px",
      },
    },
  },
  plugins: [],
};
