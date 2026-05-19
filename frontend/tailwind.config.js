/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Star-gold accent — like distant suns
        brand: {
          50:  '#fffbeb',
          100: '#fef3c7',
          400: '#fbbf24',
          500: '#f59e0b',
          600: '#d97706',
          700: '#b45309',
        },
        // True-black space surfaces — no blue tint
        surface: {
          900: '#030305',
          800: '#0a0a0f',
          700: '#141420',
          600: '#252535',
          400: '#7878a0',
          200: '#c8c8e0',
          50:  '#eeeef8',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
}
