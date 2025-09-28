// import { Link } from "react-router-dom";
// import Spline from "@splinetool/react-spline";
// import "../styles/stars.css"; // stars effect ke liye

// export default function Home() {
//   return (
//     <main className="w-full h-screen relative">
//       {/* Background */}
//       <Spline scene="/scene.splinecode" />

//       {/* Center Button */}
//       <div className="absolute inset-0 flex justify-center items-center">
//         <Link to="/chat">
//           <button className="relative px-10 py-6 text-2xl font-bold text-white 
//             bg-black/70 rounded-2xl transition-all duration-500 
//             hover:bg-black hover:scale-110 overflow-hidden">
            
//             {/* Stars effect */}
//             <span className="stars"></span>
//             <span className="relative z-10">🚀 Start Chat</span>
//           </button>
//         </Link>
//       </div>
//     </main>
//   );
// }

// import React from 'react';
// import { useNavigate } from 'react-router-dom';
// import Spline from '@splinetool/react-spline';
// import AIBot from '../components/AIBot';   // ✅ corrected path
// import '../styles/stars.css';              // ✅ corrected path

// export default function Home() {
//   const navigate = useNavigate();

//   return (
//     <main className="w-screen h-screen overflow-hidden">
//       <section className="w-full h-full relative flex items-center justify-center">
//         <Spline scene="/scene.splinecode" className="absolute inset-0 z-0" />
//         <button
//           onClick={() => navigate('/chat')}
//           className="relative px-8 py-4 text-white text-lg font-semibold rounded-lg bg-transparent border-2 border-white overflow-hidden transition-all duration-300 ease-in-out group z-10"
//         >
//           <span className="relative z-10">Enter AI Chatbot</span>
//           <div className="absolute inset-0 bg-black opacity-0 group-hover:opacity-100 transition-opacity duration-300">
//             <div className="absolute inset-0 animate-twinkle">
//               {[...Array(20)].map((_, i) => (
//                 <div
//                   key={i}
//                   className="absolute bg-white rounded-full"
//                   style={{
//                     width: Math.random() * 3 + 1 + 'px',
//                     height: Math.random() * 3 + 1 + 'px',
//                     top: Math.random() * 100 + '%',
//                     left: Math.random() * 100 + '%',
//                     animation: `twinkle ${Math.random() * 2 + 1}s infinite`,
//                   }}
//                 />
//               ))}
//             </div>
//           </div>
//         </button>
//       </section>
//     </main>
//   );
// }


import React from 'react';
import { useNavigate } from 'react-router-dom';
import Spline from '@splinetool/react-spline';
import '../styles/stars.css'; // Correct path

export default function Home() {
  const navigate = useNavigate();

  return (
    <main className="w-screen h-screen overflow-hidden">
      <section className="w-full h-full relative flex items-center justify-center">
        <Spline scene="/scene.splinecode" className="absolute inset-0 z-0" />
        <button
          onClick={() => navigate('/chat')}
          className="relative px-8 py-4 text-white text-lg font-semibold rounded-lg bg-transparent border-2 border-white overflow-hidden transition-all duration-300 ease-in-out group z-10"
        >
          <span className="relative z-10">Start Flood Forecasting AI</span>
          <div className="absolute inset-0 bg-black opacity-0 group-hover:opacity-80 transition-opacity duration-300 z-0">
            <div className="stars z-1"></div>
          </div>
        </button>
      </section>
    </main>
  );
}