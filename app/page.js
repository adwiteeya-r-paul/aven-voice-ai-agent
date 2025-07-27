'use client';

import Script from 'next/script';

export default function Home() {
  return (
    <div>
      <main>
        <h1 style={{ textAlign: 'center' }}>Welcome to Aven Customer Support!</h1>
        <p style={{ textAlign: 'center' }}>Click the Vapi widget below to start a conversation with Abby.</p>
      </main>

      <vapi-widget
        assistant-id="05fc5585-41b3-479e-a6c4-979a824aec23"
        public-key="5aa52943-5448-433d-999f-0dc4e5b7ab5a"
      ></vapi-widget>

      <Script
        src="https://unpkg.com/@vapi-ai/client-sdk-react/dist/embed/widget.umd.js"
        async
        strategy="lazyOnload"
        onLoad={() => {
          console.log("Vapi SDK Script loaded successfully from unpkg.");
        }}
        onError={(e) => console.error("Error loading Vapi SDK script from unpkg:", e)}
      />
    </div>
  );
}
