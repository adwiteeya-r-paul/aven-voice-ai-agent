'use client';

import Script from 'next/script';
import { useEffect, useState } from 'react';

export default function Home() {
  const [vapiLoaded, setVapiLoaded] = useState(false);

  useEffect(() => {
    const checkVapi = setInterval(() => {
      if (typeof window !== 'undefined' && typeof window.Vapi !== 'undefined' && !vapiLoaded) {
        console.log("Vapi object detected, initializing listen...");
        window.Vapi.listen({
          assistantId: "05fc5585-41b3-479e-a6c4-979a824aec23",
          button: true
        });
        setVapiLoaded(true);
        clearInterval(checkVapi);
      }
    }, 200);

    return () => clearInterval(checkVapi);
  }, [vapiLoaded]);

  return (
    <div>
      <main>
        <h1 style={{ textAlign: 'center' }}>Welcome to Aven Customer Support!</h1>
        <p style={{ textAlign: 'center' }}>Click the Vapi widget below to start a conversation with Abby.</p>
      </main>

      <Script
        src="https://vapi.ai/widgets/v1/script.js"
        strategy="lazyOnload"
        onLoad={() => {
          console.log("Vapi Script loaded via onLoad callback.");
        }}
        onError={(e) => console.error("Error loading Vapi script:", e)}
      />
    </div>
  );
}
