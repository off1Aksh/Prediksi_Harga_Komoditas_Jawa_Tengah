const Footer = () => {
  return (
    <footer className="bg-white border-t border-emerald-50 py-8 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col items-center justify-center text-center">
          <p className="text-gray-500 text-sm">
            &copy; {new Date().getFullYear()} PanganPredict. Skripsi Project. All rights reserved.
          </p>
          <p className="text-emerald-600 text-xs mt-2 font-medium">
            Univariate LSTM Architecture
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
