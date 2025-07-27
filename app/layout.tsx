// app/layout.tsx

import './globals.css'
import { Inter } from 'next/font/google'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Abby is here to support you!', // <--- UPDATE THIS LINE
  description: 'Aven Voice AI Agent powered by Next.js and Flask', // <--- ADD/UPDATE THIS LINE
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  )
}
