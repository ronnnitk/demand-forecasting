import React from 'react';
import { motion } from 'framer-motion';

interface ScrollContainerProps {
  children: React.ReactNode;
}

const ScrollContainer: React.FC<ScrollContainerProps> = ({ children }) => {
  return (
    <motion.div
      className="min-h-screen"
      initial={false}
      animate={{ opacity: 1, y: 0 }}
    >
      {children}
    </motion.div>
  );
};

export default ScrollContainer;