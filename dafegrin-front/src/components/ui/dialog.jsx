import React from "react";

export const Dialog = ({ open, onOpenChange, children }) => {
  if (!open) return null;
  return (
    <div
      className="fixed inset-0 bg-black/30 flex justify-center items-center"
      onClick={() => onOpenChange(false)}
    >
      <div className="bg-white p-6 rounded" onClick={(e) => e.stopPropagation()}>
        {children}
      </div>
    </div>
  );
};

export const DialogContent = ({ children }) => <div>{children}</div>;
export const DialogTitle = ({ children }) => (
  <h2 className="text-lg font-bold mb-2">{children}</h2>
);
