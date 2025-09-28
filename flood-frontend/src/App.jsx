// // import Spline from '@splinetool/react-spline';
// // import AIBot from './components/AIBot';
// // import './styles/aibot.css';

// // export default function App() {
// //   return (
// //     <main className="w-full h-full">
// //       {/* Fullscreen Spline Scene */}
// //       <section className="h-screen w-full">
// //         <Spline scene="/scene.splinecode" />
// //       </section>

// //       {/* AI Bot Section */}
// //       <AIBot />
// //     </main>
// //   );
// // }


// import Spline from '@splinetool/react-spline';
// import AIBot from './components/AIBot';
// import './styles/aibot.css';

// export default function App() {
//   return (
//     <main className="w-full h-full">
//       {/* Fullscreen Spline Scene as Background */}
//       <section className="w-full h-full">
//         <Spline scene="/scene.splinecode" />
//       </section>

//       {/* AI Bot Section Full Page */}
//       <div className="w-full h-full">
//         <AIBot />
//       </div>
//     </main>
//   );
// }


import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import AIBot from "./components/AIBot";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/chat" element={<AIBot />} />
      </Routes>
    </Router>
  );
}