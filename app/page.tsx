// app/page.js

import Script from 'next/script'; 

export const metadata = {
  title: 'Abby is here to support you!',
};

export default function Home() {
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
          if (typeof Vapi !== 'undefined') { 
            Vapi.listen({
              assistantId: "05fc5585-41b3-479e-a6c4-979a824aec23",
              button: true
            });
          } else {
            console.error("Vapi script loaded, but Vapi object not found.");
          }
        }}
      />
    </div>
  );
}
