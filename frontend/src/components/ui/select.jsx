import React from 'react';
import { cn } from '@/lib/utils';

const Select = ({ children, value, onValueChange }) => {
  return (
    <div className="relative">
      {React.Children.map(children, child => 
        React.cloneElement(child, { value, onValueChange })
      )}
    </div>
  );
};

const SelectTrigger = React.forwardRef(({ className, children, value, onValueChange, ...props }, ref) => (
  <button
    ref={ref}
    className={cn(
      "flex h-10 w-full items-center justify-between rounded-md border border-gray-300 bg-white px-3 py-2 text-sm ring-offset-white placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
      className
    )}
    {...props}
  >
    {children}
  </button>
));
SelectTrigger.displayName = "SelectTrigger";

const SelectValue = ({ placeholder, value }) => (
  <span className={value ? "text-gray-900" : "text-gray-500"}>
    {value || placeholder}
  </span>
);

const SelectContent = ({ children, value, onValueChange }) => (
  <div className="absolute top-full left-0 z-50 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg">
    {React.Children.map(children, child =>
      React.cloneElement(child, { onValueChange, currentValue: value })
    )}
  </div>
);

const SelectItem = ({ value, children, onValueChange, currentValue }) => (
  <div
    className={cn(
      "px-3 py-2 text-sm cursor-pointer hover:bg-gray-100",
      currentValue === value && "bg-blue-50 text-blue-600"
    )}
    onClick={() => onValueChange(value)}
  >
    {children}
  </div>
);

export { Select, SelectTrigger, SelectValue, SelectContent, SelectItem };