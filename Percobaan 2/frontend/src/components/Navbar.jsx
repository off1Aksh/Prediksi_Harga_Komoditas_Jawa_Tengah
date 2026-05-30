import { Link, useLocation } from 'react-router-dom';
import { Leaf } from 'lucide-react';

const Navbar = () => {
  const location = useLocation();
  const isActive = (path) => location.pathname === path;

  return (
    <nav className="bg-white shadow-sm border-b border-emerald-50 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center gap-2">
              <Leaf className="h-6 w-6 text-emerald-600" />
              <span className="font-bold text-xl text-gray-900 tracking-tight">Pangan<span className="text-emerald-600">Predict</span></span>
            </Link>
          </div>
          <div className="flex items-center space-x-6 sm:space-x-8">
            <Link 
              to="/" 
              className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors h-full mt-1
                ${isActive('/') ? 'border-emerald-500 text-emerald-700' : 'border-transparent text-gray-500 hover:text-emerald-600 hover:border-emerald-300'}`}
            >
              Home
            </Link>
            <Link 
              to="/dashboard" 
              className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors h-full mt-1
                ${isActive('/dashboard') ? 'border-emerald-500 text-emerald-700' : 'border-transparent text-gray-500 hover:text-emerald-600 hover:border-emerald-300'}`}
            >
              Dashboard
            </Link>
            <Link 
              to="/about" 
              className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors h-full mt-1
                ${isActive('/about') ? 'border-emerald-500 text-emerald-700' : 'border-transparent text-gray-500 hover:text-emerald-600 hover:border-emerald-300'}`}
            >
              About
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
