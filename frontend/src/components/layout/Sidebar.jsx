import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  Home, 
  Brain, 
  Database, 
  Settings, 
  BarChart3, 
  Leaf, 
  Zap,
  Users,
  FileText,
  Cpu
} from 'lucide-react';

const Sidebar = () => {
  const location = useLocation();

  const navigationItems = [
    {
      name: 'Dashboard',
      href: '/dashboard',
      icon: Home,
      description: 'Overview and analytics'
    },
    {
      name: 'Model Training',
      href: '/training',
      icon: Brain,
      description: 'Train AI models'
    },
    {
      name: 'Datasets',
      href: '/datasets',
      icon: Database,
      description: 'Manage training data'
    },
    {
      name: 'Carbon Analytics',
      href: '/carbon-analytics',
      icon: Leaf,
      description: 'Environmental impact'
    },
    {
      name: 'GPU Management',
      href: '/gpu-management',
      icon: Cpu,
      description: 'Hardware resources'
    },
    {
      name: 'Analytics',
      href: '/analytics',
      icon: BarChart3,
      description: 'Performance metrics'
    },
    {
      name: 'Team',
      href: '/team',
      icon: Users,
      description: 'Collaborate with team'
    },
    {
      name: 'Documentation',
      href: '/docs',
      icon: FileText,
      description: 'Guides and API docs'
    },
    {
      name: 'Settings',
      href: '/settings',
      icon: Settings,
      description: 'Account and preferences'
    }
  ];

  const isActive = (href) => location.pathname === href;

  return (
    <div className="w-64 bg-white border-r border-gray-200 h-screen overflow-y-auto">
      {/* Logo */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center space-x-2">
          <div className="bg-blue-600 p-2 rounded-lg">
            <Brain className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-gray-900">DeepForge</h1>
            <p className="text-xs text-gray-600">AI Builder Platform</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="p-4 space-y-2">
        {navigationItems.map((item) => {
          const Icon = item.icon;
          const active = isActive(item.href);
          
          return (
            <Link
              key={item.name}
              to={item.href}
              className={`flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors group ${
                active
                  ? 'bg-blue-50 text-blue-700 border-r-2 border-blue-600'
                  : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
              }`}
            >
              <Icon className={`h-5 w-5 ${
                active ? 'text-blue-600' : 'text-gray-400 group-hover:text-gray-600'
              }`} />
              <div className="flex-1">
                <div className="font-medium text-sm">{item.name}</div>
                <div className="text-xs text-gray-500">{item.description}</div>
              </div>
              
              {/* Special indicators */}
              {item.name === 'Carbon Analytics' && (
                <div className="bg-green-100 text-green-600 text-xs px-2 py-1 rounded-full">
                  New
                </div>
              )}
              
              {item.name === 'Model Training' && active && (
                <Zap className="h-3 w-3 text-yellow-500" />
              )}
            </Link>
          );
        })}
      </nav>

      {/* Carbon Impact Summary (Mini Widget) */}
      <div className="p-4 border-t border-gray-200 mt-auto">
        <div className="bg-green-50 p-3 rounded-lg">
          <div className="flex items-center space-x-2 mb-2">
            <Leaf className="h-4 w-4 text-green-600" />
            <span className="text-sm font-medium text-green-900">Today's Impact</span>
          </div>
          <div className="text-xs text-green-700 space-y-1">
            <div>CO₂ Saved: 2.3 kg</div>
            <div>Efficient GPU Usage: 87%</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;