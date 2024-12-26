'use client'

import { useState, useEffect } from 'react'
import { IntroSlide } from './components/intro-slide'
import { PhotoGrid } from './components/photo-grid'

export default function Page() {
  const [showIntro, setShowIntro] = useState(true)

  useEffect(() => {
    const handleInteraction = () => {
      setShowIntro(false)
    }

    if (showIntro) {
      window.addEventListener('keydown', handleInteraction)
      window.addEventListener('click', handleInteraction)
    }

    return () => {
      window.removeEventListener('keydown', handleInteraction)
      window.removeEventListener('click', handleInteraction)
    }
  }, [showIntro])

  return (
    <main className="min-h-screen bg-gradient-to-b from-red-700 to-green-800">
      {showIntro ? <IntroSlide /> : <PhotoGrid />}
    </main>
  )
}

