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
        public-key="5aa52943-5448-433d-999f-0dc4e5b7ab5a"
        assistant-id="05fc5585-41b3-479e-a6c4-979a824aec23"
        mode="voice"
        theme="dark"
        base-bg-color="#000000"
        accent-color="#14B8A6"
        cta-button-color="#000000"
        cta-button-text-color="#ffffff"
        border-radius="large"
        size="full"
        position="bottom-right"
        title="TALK WITH AI"
        start-button-text="Start"
        end-button-text="End Call"
        chat-first-message="Hey, How can I help you today?"
        chat-placeholder="Type your message..."
        voice-show-transcript="true"
        consent-required="true"
        consent-title="Terms and conditions"
        consent-content="By clicking \"Agree,\" and each time I interact with this AI agent, I consent to the recording, storage, and sharing of my communications with third-party service providers, and as otherwise described in our Terms of Service."
        consent-storage-key="vapi_widget_consent"
      ></vapi-widget>

      <Script
        src="https://unpkg.com/@vapi-ai/client-sdk-react/dist/embed/widget.umd.js"
        async
        strategy="lazyOnload"
      />
    </div>
  );
}

