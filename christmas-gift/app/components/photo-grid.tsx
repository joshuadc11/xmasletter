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
    poem: 'Poem 1\n\nReplace this with your actual poem text.' 
  },
  { 
    id: 2, 
    title: 'Flower', 
    imageUrl: '/flowerimage.jpeg', 
    poem: 'Poem 2\n\nReplace this with your actual poem text.' 
  },
  { 
    id: 3, 
    title: 'Color', 
    imageUrl: '/colorimage.jpeg', 
    poem: 'Poem 3\n\nReplace this with your actual poem text.' 
  },
  { 
    id: 4, 
    title: 'Season', 
    imageUrl: '/seasonimage.jpeg', // You'll need to provide this image
    poem: 'Poem 4\n\nReplace this with your actual poem text.' 
  },
  { 
    id: 5, 
    title: 'Character', 
    imageUrl: '/characterimage.jpeg', 
    poem: 'Poem 5\n\nReplace this with your actual poem text.' 
  },
  { 
    id: 6, 
    title: 'City', 
    imageUrl: '/cityimage.jpeg', 
    poem: 'Poem 6\n\nReplace this with your actual poem text.' 
  },
  { 
    id: 7, 
    title: 'Food', 
    imageUrl: '/foodimage.jpeg', 
    poem: 'Poem 7\n\nReplace this with your actual poem text.' 
  },
  { 
    id: 8, 
    title: 'Place', 
    imageUrl: '/placeimage.jpeg', 
    poem: 'Poem 8\n\nReplace this with your actual poem text.' 
  },
  { 
    id: 9, 
    title: 'Animal', 
    imageUrl: '/animalimage.jpeg', 
    poem: 'Poem 9\n\nReplace this with your actual poem text.' 
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

