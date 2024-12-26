'use client'

import { motion } from 'framer-motion'

interface IntroSlideProps {
  onComplete: () => void
}

export function IntroSlide({ onComplete }: IntroSlideProps) {

  return (
    <motion.div
      className="flex min-h-screen items-center justify-center px-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <h1 className="text-center font-cursive text-2xl text-white md:text-4xl lg:text-5xl">
        Merry Christmas my <span className="font-bold">Dear Angel</span>,
        <br />
        to another amazing year with my <span className="font-bold">Best Friend</span>.
      </h1>
    </motion.div>
  )
}

