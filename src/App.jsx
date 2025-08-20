import React from 'react';
import Navbar from './components/layouts/navbar';
import HeroSection from './components/sections/HeroSection';
import FeatureCards from './components/sections/FeatureCards';
import AdditionalFeatures from './components/sections/AdditionalFeatures';
import Footer from './components/layouts/Footer';
import { useScrollAnimation } from './hooks/useScrollAnimation';

function App() {
  // Initialize scroll animations
  useScrollAnimation();

  return (
    <div className="App">
      <Navbar />
      <HeroSection />
      <FeatureCards />
      <AdditionalFeatures />
      <Footer />
    </div>
  );
}

export default App;