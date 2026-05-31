"use client";

import { Filter, Calendar, Trophy, Activity, Zap } from 'lucide-react';

interface FilterState {
  date: string;
  league: string;
  status: 'all' | 'scheduled' | 'live' | 'finished';
  showTop7Only: boolean;
  sortBy: 'confidence' | 'time' | 'league';
}

interface MatchFiltersProps {
  filters: FilterState;
  onFilterChange: (filters: FilterState) => void;
  totalMatches: number;
}

export default function MatchFilters({ filters, onFilterChange, totalMatches }: MatchFiltersProps) {
  const updateFilter = (key: keyof FilterState, value: any) => {
    onFilterChange({ ...filters, [key]: value });
  };

  return (
    <div className="bg-gradient-to-br from-slate-800/90 via-slate-800/80 to-slate-900/90 backdrop-blur-xl rounded-2xl border border-slate-700/30 shadow-2xl mb-8 overflow-hidden">
      {/* Header Bar with Gradient */}
      <div className="bg-gradient-to-r from-blue-600/10 via-purple-600/10 to-blue-600/10 border-b border-slate-700/30 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-500/10 rounded-lg border border-blue-500/20">
              <Filter size={20} className="text-blue-400" />
            </div>
            <div>
              <h3 className="text-white font-bold text-lg">Smart Filters</h3>
              <p className="text-slate-400 text-xs">Refine your match selection</p>
            </div>
          </div>

          {/* Results Counter */}
          <div className="px-4 py-2 bg-slate-900/50 rounded-lg border border-slate-700/50">
            <div className="flex items-center gap-2">
              <span className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                {totalMatches}
              </span>
              <span className="text-slate-400 text-sm">matches</span>
            </div>
            {filters.showTop7Only && (
              <p className="text-xs text-orange-400 font-medium mt-0.5">Top 7 Scientific Picks</p>
            )}
          </div>
        </div>
      </div>

      <div className="p-6 space-y-5">
        {/* Date & League Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Date Filter */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
              <Calendar size={14} />
              Match Date
            </label>
            <div className="flex items-center gap-2">
              <div className="relative flex-1">
                <Calendar size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none z-10" />
                <input
                  type="date"
                  value={
                    filters.date === 'today'
                      ? (() => { 
                          const d = new Date(); 
                          const brt = new Date(d.getTime() - 3 * 60 * 60 * 1000); 
                          return brt.toISOString().split('T')[0]; 
                        })()
                      : filters.date === 'tomorrow'
                        ? (() => { 
                            const d = new Date(); 
                            const brt = new Date(d.getTime() - 3 * 60 * 60 * 1000 + 24 * 60 * 60 * 1000); 
                            return brt.toISOString().split('T')[0]; 
                          })()
                        : filters.date
                  }
                  onChange={(e) => updateFilter('date', e.target.value)}
                  className="w-full appearance-none bg-slate-900/80 border border-slate-700/50 rounded-xl pl-11 pr-4 py-3 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all cursor-pointer hover:border-slate-600"
                />
              </div>

              {/* Quick Filters */}
              <div className="flex items-center gap-1.5 bg-slate-900/50 border border-slate-700/50 rounded-xl p-1">
                <button
                  onClick={() => updateFilter('date', 'today')}
                  className={`px-3.5 py-2 text-xs font-semibold rounded-lg transition-all ${filters.date === 'today'
                    ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-lg shadow-blue-500/30'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                    }`}
                >
                  Today
                </button>
                <button
                  onClick={() => updateFilter('date', 'tomorrow')}
                  className={`px-3.5 py-2 text-xs font-semibold rounded-lg transition-all ${filters.date === 'tomorrow'
                    ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-lg shadow-blue-500/30'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                    }`}
                >
                  Tomorrow
                </button>
              </div>
            </div>
          </div>

          {/* League Filter */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
              <Trophy size={14} />
              Competition
            </label>
            <div className="relative">
              <select
                value={filters.league}
                onChange={(e) => updateFilter('league', e.target.value)}
                className="w-full appearance-none bg-slate-900/80 border border-slate-700/50 rounded-xl px-4 py-3 pr-11 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all cursor-pointer hover:border-slate-600"
              >
                <option value="all">🏆 All Leagues</option>
                <option value="premier_league">Premier League</option>
                <option value="la_liga">La Liga</option>
                <option value="serie_a">Serie A</option>
                <option value="bundesliga">Bundesliga</option>
                <option value="ligue1">Ligue 1</option>
                <option value="brasileirao_a">Brasileirão A</option>
                <option value="brasileirao_b">Brasileirão B</option>
              </select>
              <Trophy size={16} className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
            </div>
          </div>
        </div>

        {/* Status Filter */}
        <div className="space-y-2">
          <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
            <Activity size={14} />
            Match Status
          </label>
          <div className="grid grid-cols-4 gap-2">
            <button
              onClick={() => updateFilter('status', 'all')}
              className={`px-4 py-3 text-sm font-semibold rounded-xl transition-all ${filters.status === 'all'
                ? 'bg-gradient-to-br from-slate-700 to-slate-600 text-white shadow-lg border border-slate-500/50'
                : 'bg-slate-900/50 text-slate-400 hover:text-white hover:bg-slate-800 border border-slate-700/30'
                }`}
            >
              All Matches
            </button>
            <button
              onClick={() => updateFilter('status', 'scheduled')}
              className={`px-4 py-3 text-sm font-semibold rounded-xl transition-all flex items-center justify-center gap-2 ${filters.status === 'scheduled'
                ? 'bg-gradient-to-br from-blue-600 to-blue-500 text-white shadow-lg shadow-blue-500/30 border border-blue-400/50'
                : 'bg-slate-900/50 text-slate-400 hover:text-white hover:bg-slate-800 border border-slate-700/30'
                }`}
            >
              <span className="w-2 h-2 bg-blue-400 rounded-full"></span>
              Scheduled
            </button>
            <button
              onClick={() => updateFilter('status', 'live')}
              className={`px-4 py-3 text-sm font-semibold rounded-xl transition-all flex items-center justify-center gap-2 ${filters.status === 'live'
                ? 'bg-gradient-to-br from-emerald-600 to-emerald-500 text-white shadow-lg shadow-emerald-500/30 border border-emerald-400/50'
                : 'bg-slate-900/50 text-slate-400 hover:text-white hover:bg-slate-800 border border-slate-700/30'
                }`}
            >
              <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span>
              Live Now
            </button>
            <button
              onClick={() => updateFilter('status', 'finished')}
              className={`px-4 py-3 text-sm font-semibold rounded-xl transition-all flex items-center justify-center gap-2 ${filters.status === 'finished'
                ? 'bg-gradient-to-br from-slate-700 to-slate-600 text-white shadow-lg border border-slate-500/50'
                : 'bg-slate-900/50 text-slate-400 hover:text-white hover:bg-slate-800 border border-slate-700/30'
                }`}
            >
              <span className="w-2 h-2 bg-slate-400 rounded-full"></span>
              Finished
            </button>
          </div>
        </div>

        {/* Bottom Row - Sort & Top 7 */}
        <div className="flex items-center justify-between gap-4 pt-3 border-t border-slate-700/30">
          {/* Top 7 Toggle */}
          <button
            onClick={() => updateFilter('showTop7Only', !filters.showTop7Only)}
            className={`flex-1 max-w-xs px-5 py-3 rounded-xl text-sm font-bold flex items-center justify-center gap-2.5 transition-all ${filters.showTop7Only
              ? 'bg-gradient-to-r from-orange-600 via-red-600 to-orange-600 text-white shadow-xl shadow-orange-500/40 border border-orange-400/50'
              : 'bg-slate-900/50 text-slate-400 border border-slate-700/50 hover:border-orange-500/50 hover:text-orange-400'
              }`}
          >
            <Zap size={18} className={filters.showTop7Only ? 'animate-pulse' : ''} />
            {filters.showTop7Only ? 'Top 7 Scientific' : 'Show Top 7 Picks'}
          </button>

          {/* Sort By */}
          <div className="flex-1 max-w-xs relative">
            <select
              value={filters.sortBy}
              onChange={(e) => updateFilter('sortBy', e.target.value as any)}
              className="w-full appearance-none bg-slate-900/80 border border-slate-700/50 rounded-xl px-4 py-3 pr-11 text-sm font-medium text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all cursor-pointer hover:border-slate-600"
            >
              <option value="confidence">⚡ Sort by: Confidence</option>
              <option value="time">🕐 Sort by: Time</option>
              <option value="league">🏆 Sort by: League</option>
            </select>
            <Activity size={16} className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
          </div>
        </div>
      </div>
    </div>
  );
}
