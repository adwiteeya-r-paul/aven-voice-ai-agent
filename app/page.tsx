// app/page.js

import Script from 'next/script'; 

export const metadata = {
  title: 'Abby is here to support you!',
};

export default function Home() {
  return (
    <div>
     
      <main>
        <h1>Welcome to Aven Customer Support!</h1>
        <p>Click the Vapi widget below to start a conversation with Abby.</p>
      </main>

      {/* VAPI WIDGET SCRIPTS */}
      <Script src="https://vapi.ai/widgets/v1/script.js" strategy="lazyOnload" />
      <Script strategy="lazyOnload" id="vapi-init-script">
        {`
          Vapi.listen({
              assistantId: "05fc5585-41b3-479e-a6c4-979a824aec23",
              button: true
          });
        `}
      </Script>
    </div>
  );
}
