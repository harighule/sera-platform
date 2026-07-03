import { useEffect, useState } from 'react'

export default function AnimatedCounter({ value, duration = 1200 }) {
  const [displayValue, setDisplayValue] = useState(0)

  useEffect(() => {
    if (typeof value !== 'number') {
      setDisplayValue(value)
      return
    }

    let start = 0
    const end = value
    if (start === end) {
      setDisplayValue(end)
      return
    }

    const startTime = performance.now()

    const animate = (now) => {
      const elapsed = now - startTime
      const progress = Math.min(elapsed / duration, 1)
      
      // Easing function: easeOutQuad
      const easedProgress = progress * (2 - progress)
      
      const current = Math.floor(start + (end - start) * easedProgress)
      setDisplayValue(current)

      if (progress < 1) {
        requestAnimationFrame(animate)
      } else {
        setDisplayValue(end)
      }
    }

    requestAnimationFrame(animate)
  }, [value, duration])

  return <span>{displayValue}</span>
}
