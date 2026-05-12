"use client";

import { useState, useEffect } from 'react';
import { useLabs } from '../contexts/LabsContext';
import { FlaskConical, Zap } from 'lucide-react';
import SystemStatus from '../components/SystemStatus';
import MatchFilters from '../components/MatchFilters';
import MatchCardV2 from '../components/MatchCardV2';
import BettingSlip from '../components/BettingSlip';
import AIPerformance from '../components/AIPerformance';
import MyBets from '../components/MyBets';
import TopOpportunitiesList from '../components/TopOpportunitiesList';
import ScannerControls from '../components/ScannerControls';
import Login from '../components/Login';
import Leaderboard from '../components/Leaderboard';

interface FilterState {
  date: string;
  league: string;
  status: 'all' | 'scheduled' | 'live' | 'finished';
  showTop7Only: boolean;
  sortBy: 'confidence' | 'time' | 'league';
}

interface User {
  id: number;
  username: string;
}

export default function Dashboard() {
  const [user, setUser] = useState<User | null>(null);
  const [activeTab, setActiveTab] = useState<'scanner' | 'performance' | 'bankroll' | 'league' | 'operations'>('scanner');
  const [filters, setFilters] = useState<FilterState>({
    date: 'today',
    league: 'all',
    status: 'all',
    showTop7Only: false,
    sortBy: 'confidence'
  });

  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(true);

  // Auth Check
  useEffect(() => {
    const storedId = localStorage.getItem('userId');
    const storedName = localStorage.getItem('username');
    if (storedId && storedName) {
      setUser({ id: parseInt(storedId), username: storedName });
    }
  }, []);

  const handleLogin = (userData: User) => {
    setUser(userData);
  };
  
  const handleLogout = () => {
    localStorage.removeItem('userId');
    localStorage.removeItem('username');
    setUser(null);
  };

  useEffect(() => {
    if (activeTab === 'scanner' && user) {
      fetchMatches();
    }
  }, [filters, activeTab, user]);

  // Auto-refresh for live updates (every 60 seconds)
  useEffect(() => {
    if (activeTab !== 'scanner' || !user) return;

    const interval = setInterval(() => {
      // Only refresh if there are live or scheduled matches
      const hasLiveOrScheduled = matches.some(
        (m: any) => m.status === 'inprogress' || m.status === 'notstarted' || m.status === 'scheduled'
      );
      
      if (hasLiveOrScheduled) {
        console.log('🔄 Auto-refreshing live matches...');
        fetchMatches();
      }
    }, 60000); // 60 seconds

    return () => clearInterval(interval);
  }, [activeTab, matches, user]);

  const fetchMatches = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/predictions?' + new URLSearchParams({
        type: 'predictions',
        date: filters.date,
        league: filters.league,
        status: filters.status,
        top7Only: filters.showTop7Only.toString(),
        sortBy: filters.sortBy
      }));
      const data = await response.json();
      setMatches(data.matches || []);
    } catch (error) {
      console.error('Failed to fetch matches:', error);
    } finally {
      setLoading(false);
    }
  };

  if (!user) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col">
      {/* Header */}
      <header className="bg-slate-900 border-b border-slate-700/50 sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-4 lg:px-6 py-3">
          {/* Top Row: Branding + User Info */}
          <div className="flex items-center justify-between mb-3">
            {/* Left: Logo + Status */}
            <div className="flex items-center gap-4 lg:gap-6">
              <div className="flex items-center gap-3">
                <h1 className="text-xl lg:text-2xl font-bold text-white tracking-tight">Cortex AI</h1>
                <div className="px-2 py-0.5 bg-blue-600 rounded text-[10px] font-bold text-white uppercase tracking-wider">
                  V6 Pro
                </div>
              </div>
              
              <div className="hidden md:flex items-center gap-4">
                <SystemStatus mode="compact" />
                
                {/* LABS TOGGLE (Hardcoded into page header) */}
                <ShadowModeToggle />
              </div>
            </div>

            {/* Right: User Profile & Stats */}
            <div className="flex items-center gap-4 lg:gap-6 text-sm">
               <div className="flex items-center gap-2 lg:gap-3 bg-slate-800 px-2.5 lg:px-3 py-1.5 rounded-full border border-slate-700">
                 <span className="hidden sm:inline text-slate-400 text-xs uppercase font-bold">User</span>
                 <span className="text-white font-bold text-sm">{user.username}</span>
                 <button onClick={handleLogout} className="text-red-400 hover:text-red-300 text-xs font-medium">Logout</button>
               </div>
               
              <div className="hidden xl:flex items-center gap-4">
                <div className="w-px h-8 bg-slate-700" />
                <div className="flex flex-col items-end">
                  <span className="text-slate-500 text-[10px] uppercase font-bold tracking-wide">Top 7 Health</span>
                  <span className="text-emerald-400 font-bold text-sm">82.0% Win Rate</span>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Row: Navigation */}
          <nav className="flex items-center gap-1 bg-slate-950/50 p-1 rounded-lg border border-slate-800 overflow-x-auto scrollbar-hide">
            <button
              onClick={() => setActiveTab('scanner')}
              className={`px-3 lg:px-4 py-2 text-xs lg:text-sm font-medium rounded-md transition-all whitespace-nowrap ${
                activeTab === 'scanner' 
                  ? 'bg-blue-600 text-white shadow-lg' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              Scanner
            </button>
            <button
              onClick={() => setActiveTab('performance')}
              className={`px-3 lg:px-4 py-2 text-xs lg:text-sm font-medium rounded-md transition-all whitespace-nowrap ${
                activeTab === 'performance' 
                  ? 'bg-blue-600 text-white shadow-lg' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              AI Performance
            </button>
            <button
              onClick={() => setActiveTab('bankroll')}
              className={`px-3 lg:px-4 py-2 text-xs lg:text-sm font-medium rounded-md transition-all whitespace-nowrap ${
                activeTab === 'bankroll' 
                  ? 'bg-blue-600 text-white shadow-lg' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              My Bets
            </button>
            <button
              onClick={() => setActiveTab('league')}
              className={`px-3 lg:px-4 py-2 text-xs lg:text-sm font-medium rounded-md transition-all whitespace-nowrap ${
                activeTab === 'league' 
                  ? 'bg-blue-600 text-white shadow-lg' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              Tipster League
            </button>
            <button
              onClick={() => setActiveTab('operations')}
              className={`px-3 lg:px-4 py-2 text-xs lg:text-sm font-medium rounded-md transition-all whitespace-nowrap ${
                activeTab === 'operations' 
                  ? 'bg-blue-600 text-white shadow-lg' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              Operações
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl mx-auto px-4 md:px-6 py-8 w-full lg:pr-[400px] pb-24 lg:pb-8">
        {activeTab === 'scanner' && (
          <div className="animate-in fade-in slide-in-from-bottom-2 duration-500">
            <ScannerControls onScanComplete={fetchMatches} />
            
            <MatchFilters 
              filters={filters}
              onFilterChange={setFilters}
              totalMatches={matches.length}
            />

            <div className="space-y-6">
              {loading ? (
                <div className="flex flex-col items-center justify-center py-32 space-y-4">
                  <div className="animate-spin rounded-full h-10 w-10 border-4 border-blue-600 border-t-transparent shadow-blue-500/20 shadow-2xl"></div>
                  <p className="text-slate-500 text-sm animate-pulse">Scanning database for premium opportunities...</p>
                </div>
              ) : matches.length === 0 ? (
                <div className="text-center py-32 bg-slate-900/30 rounded-3xl border border-dashed border-slate-800">
                  <div className="text-6xl mb-6 grayscale opacity-20">🎯</div>
                  <h3 className="text-white text-xl font-bold mb-2">No Matches Found</h3>
                  <p className="text-slate-500 text-sm max-w-xs mx-auto">
                    We couldn't find any matches matching your filters in today's database.
                  </p>
                </div>
              ) : (
                matches.map((match: any, index: number) => (
                  <MatchCardV2 key={match.id} match={match} rank={index + 1} />
                ))
              )}
            </div>
          </div>
        )}

        {activeTab === 'performance' && (
          <div className="animate-in fade-in duration-500">
            <AIPerformance />
          </div>
        )}

        {activeTab === 'bankroll' && (
          <div className="animate-in fade-in duration-500">
            <MyBets />
          </div>
        )}

        {activeTab === 'league' && (
          <div className="animate-in fade-in duration-500">
            <Leaderboard />
          </div>
        )}

        {activeTab === 'operations' && (
          <div className="animate-in fade-in duration-500">
            <div className="bg-slate-900/40 rounded-2xl border border-slate-800 p-4">
              <h3 className="text-white font-bold mb-3">Operações (Treino, Scanner e Atualização)</h3>
              <SystemStatus mode="full" />
            </div>
          </div>
        )}
      </main>

      {/* Betting Slip Sidebar */}
      <BettingSlip />
    </div>
  );
}

function ShadowModeToggle() {
  const { isShadowMode, toggleShadowMode } = useLabs();
  
  return (
    <button
      onClick={toggleShadowMode}
      className={`relative flex items-center gap-2 px-3 py-1.5 rounded-md border transition-all ${
        isShadowMode 
          ? "bg-purple-900/40 border-purple-500/50 text-purple-200 shadow-[0_0_15px_rgba(168,85,247,0.3)]" 
          : "bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-500 hover:text-white"
      }`}
    >
      <FlaskConical size={14} className={isShadowMode ? "text-purple-400 fill-purple-400/20" : ""} />
      <span className="text-[10px] font-bold uppercase tracking-wider">
        {isShadowMode ? "Ghost ON" : "Labs"}
      </span>
      {isShadowMode && (
         <span className="absolute -top-1 -right-1 flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-purple-500"></span>
         </span>
      )}
    </button>
  );
}
