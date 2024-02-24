import React, { useState } from "react";

export default function CardsContainer() {
  const [facilityRecords, setFacilityRecords] = useState(0);
  const [hdssRecords, setHdssRecords] = useState(0);
  const [showHdssModal, setShowHdssModal] = useState(false);
  const [showFacilityModal, setShowModal] = useState(false);

  const handleFileUpload = (event, setter) => {
    const file = event.target.files[0];
    if (file) {
      // Process the file and update the record count
      // For example, setFacilityRecords(100);
      setter(100); // Placeholder value
    }
  };

  return (
    <div>
      <div className="cards-container">
        <div>
          <a href="#" onClick={() => setShowModal(true)}>
            Upload Facility Data
          </a>
          <div className="card d-flex justify-content-center align-items-center">
            <h2>Facility Data</h2>
            <p>Number of records: {facilityRecords}</p>
          </div>
        </div>
        <div>
          <a href="#" onClick={() => setShowModal(true)}>
            Update HDSS Data
          </a>
          <div className="card d-flex justify-content-center align-items-center">
            <h2>HDSS Data</h2>
            <p>Number of records: {hdssRecords}</p>
          </div>
        </div>
      </div>
      {showFacilityModal && (
        <div className="modal-overlay">
          <div className="custom-modal">
            <input
              type="file"
              onChange={(e) => handleFileUpload(e, setFacilityRecords)}
            />
            <button onClick={() => setShowModal(false)}>Close</button>
          </div>
        </div>
      )}

      {showHdssModal && (
        <div className="custom-modal">
          <input
            type="file"
            onChange={(e) => handleFileUpload(e, setFacilityRecords)}
          />
          <button onClick={() => setShowModal(false)}>Close</button>
        </div>
      )}
    </div>
  );
}
