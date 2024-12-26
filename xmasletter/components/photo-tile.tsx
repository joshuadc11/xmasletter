'use client'

import { motion } from 'framer-motion'
import Image from 'next/image'

interface PhotoTileProps {
  id: number
  imageUrl: string
  poem: string
  title: string
  isExpanded: boolean
  onClick: () => void
}

export function PhotoTile({ imageUrl, poem, title, isExpanded, onClick }: PhotoTileProps) {
  return (
    <motion.div
      layout
      className={`cursor-pointer overflow-hidden rounded-lg ${
        isExpanded ? 'col-span-full row-span-full' : ''
      }`}
      onClick={onClick}
      initial={false}
      animate={{
        height: isExpanded ? 400 : 200,
      }}
      transition={{ duration: 0.3 }}
    >
      <div
        className="relative h-full w-full"
        style={{ transform: isExpanded ? 'rotateY(180deg)' : 'rotateY(0deg)', transformStyle: 'preserve-3d', transition: 'transform 0.6s' }}
      >
        <div
          className="absolute h-full w-full backface-hidden"
          style={{ backfaceVisibility: 'hidden' }}
        >
          <div className="relative h-full w-full">
            <Image
              src={imageUrl}
              alt={`${title} memory`}
              className="h-full w-full object-cover"
              width={800}
              height={800}
              priority
            />
            <div className="absolute inset-0 flex items-center justify-center bg-black/30">
              <h2 className="text-3xl font-bold text-white">{title}</h2>
            </div>
          </div>
        </div>
        <div
          className="absolute h-full w-full bg-white p-6 backface-hidden"
          style={{ backfaceVisibility: 'hidden', transform: 'rotateY(180deg)' }}
        >
          <div className="prose mx-auto max-w-none whitespace-pre-wrap text-center">
            {poem}
          </div>
        </div>
      </div>
    </motion.div>
  )
}

