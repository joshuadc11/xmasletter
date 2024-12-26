'use client'

import { useState } from 'react'
import { PhotoTile } from './photo-tile'

interface TileData {
  id: number
  imageUrl: string
  poem: string
  title: string
}

const tiles: TileData[] = [
  { 
    id: 1, 
    title: 'Hobby', 
    imageUrl: '/hobbyimage.jpeg', 
    poem: 'Poem about writing\n\nWords flow like a gentle stream,\nThoughts captured on paper gleam.\nIn quiet moments, pen in hand,\nCreating worlds at my command.' 
  },
  { 
    id: 2, 
    title: 'Flower', 
    imageUrl: '/flowerimage.jpeg', 
    poem: 'Ode to Tulips\n\nOrange petals, soft and bright,\nSpring\'s messenger, a lovely sight.\nGraceful stems reach for the sky,\nNature\'s beauty, can\'t deny.' 
  },
  { 
    id: 3, 
    title: 'Color', 
    imageUrl: '/colorimage.jpeg', 
    poem: 'Green Serenity\n\nVibrant green, like fresh spring leaves,\nCalming hue that never deceives.\nMatcha\'s essence, rich and pure,\nA color that will always endure.' 
  },
  { 
    id: 4, 
    title: 'Season', 
    imageUrl: '/seasonimage.jpeg', 
    poem: 'Autumn\'s Embrace\n\nGolden leaves dance in the breeze,\nCrisp air whispers through the trees.\nWarm colors paint the fading light,\nAutumn\'s beauty, a wondrous sight.' 
  },
  { 
    id: 5, 
    title: 'Character', 
    imageUrl: '/characterimage.jpeg', 
    poem: 'Ode to Snoopy\n\nBeagle with a heart of gold,\nImagination uncontrolled.\nOn his doghouse, dreams take flight,\nSnoopy brings pure delight.' 
  },
  { 
    id: 6, 
    title: 'City', 
    imageUrl: '/cityimage.jpeg', 
    poem: 'New York, New York\n\nSkyscrapers touch the azure sky,\nCentral Park where nature lies.\nCity of dreams, never asleep,\nMemories here, forever to keep.' 
  },
  { 
    id: 7, 
    title: 'Food', 
    imageUrl: '/foodimage.jpeg', 
    poem: 'Pad Thai Delight\n\nNoodles dance in savory sauce,\nTangy, sweet, a flavor boss.\nPeanuts crunch, lime squeezed just right,\nPad Thai, a culinary delight.' 
  },
  { 
    id: 8, 
    title: 'Place', 
    imageUrl: '/placeimage.jpeg', 
    poem: 'Coastal Haven\n\nSun-kissed shores and azure seas,\nWhispered secrets in the breeze.\nCharming town on golden sand,\nParadise in this blessed land.' 
  },
  { 
    id: 9, 
    title: 'Animal', 
    imageUrl: '/animalimage.jpeg', 
    poem: 'Manatee Magic\n\nGentle giants of the sea,\nGraceful, slow, and carefree.\nIn warm waters they gently glide,\nNature\'s beauty personified.' 
  },
]

export function PhotoGrid() {
  const [expandedTile, setExpandedTile] = useState<number | null>(null)

  return (
    <div className="container mx-auto min-h-screen p-4">
      <div
        className={`grid gap-4 ${
          expandedTile === null
            ? 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'
            : 'grid-cols-1'
        }`}
      >
        {tiles.map((tile) => (
          <PhotoTile
            key={tile.id}
            {...tile}
            isExpanded={expandedTile === tile.id}
            onClick={() => setExpandedTile(expandedTile === tile.id ? null : tile.id)}
          />
        ))}
      </div>
    </div>
  )
}

