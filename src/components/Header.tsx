import React from 'react';
import { Link } from 'react-router-dom';

const Header = () => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 flex justify-between items-center py-4 bg-white shadow-md">
      <div className="ml-6">
        <Link to="/" className="text-gray-900 text-xl font-medium">
        <img src="https://cdn.worldvectorlogo.com/logos/minecraft-1.svg" alt="Home" className="h-10 w-10 drop-shadow-lg" />
        </Link>
      </div>
      <div className="mr-6">
        <Link to="/docs" className="text-gray-900 text-xl font-medium">Docs</Link>
      </div>
    </nav>
  );
};

export default Header;
