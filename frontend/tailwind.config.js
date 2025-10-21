/** @type {import('tailwindcss').Config} */
//tells tailwind where to look for classes to include in the final CSS build
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}