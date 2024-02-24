import React, { useState } from "react";
import "./App.css";
import CardsContainer from "./components/CardsContainer";

function App() {
  
  const [activeComponent, setActiveComponent] = useState("component1");

  return (
    <div className="App">
      <header className="App-header py-5">
        <h1 className="container title-text py-5">Record Linkage in HDSS Communities within the INSPIRE Network</h1>
      </header>
      <div>
        <CardsContainer />
      </div>
      <nav className="navigation-bar">
        <a href="#" onClick={() => setActiveComponent("component1")}>
          Link 1
        </a>
        <a href="#" onClick={() => setActiveComponent("component2")}>
          Link 2
        </a>
      </nav>
      <div className="content">
        {activeComponent === "component1" ? <Component1 /> : <Component2 />}
      </div>
    </div>
  );
}

function Component1() {
  return <div>This is Component 1</div>;
}

function Component2() {
  return <div>This is Component 2</div>;
}

export default App;
