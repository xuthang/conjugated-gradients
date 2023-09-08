#include <bits/stdc++.h>

using namespace std;

typedef long long int ll; typedef vector<ll> vll;
typedef pair<ll, ll> ii; typedef vector<ii> vii;
typedef long double ld; typedef vector<ld> vld;

#define FOR(prom,a,b) for ( ll prom = (a); prom < (ll)(b); ++prom )
#define EPS (1e-10)
#define EQ(a,b) (fabs(a-b) <= fabs(a+b) * EPS)
#define PB push_back
#define deb(x) std::cerr<<#x<<" "<<x<<std::endl;
template<typename T> ostream& operator<<(ostream& out, const vector<T> & v){for(const auto & x : v){out << x << ' ';} return out;}
template<typename T> istream& operator>>(istream& in, vector<T>&v){for(auto&x:v){in>>x;}return in;}

const ll MOD = 1e9 + 7;

ll solve(string loc)
{
    ifstream in(loc);

    string infoLoc = loc.substr(0, loc.length() - loc.find_last_of('/') -1  ) + "info.txt";
    ofstream info(infoLoc);

    if(!in || !info)
        throw "file couldnt be loaded";
    
    ll n, m;
    while(!(in >> n))
    {
        in.clear();
        string tmp;
        getline(in, tmp);
        info << tmp << '\n';
    }
    info.close();

    in >> m;
    assert(n == m);

    ll amount;
    in >> amount;
    vector<pair<ii, string>> v;
    v.reserve(amount);

    ll f, s; string val;
    while(in >> f >> s >> val)
    {
        //if(EQ(val, 0)) continue;
        v.push_back({{f, s}, val});
    }

    in.close();

    ofstream out(loc);
    out << n << " " << amount << '\n';
    bool fix = v.front().first.first == 1;

    for(auto data : v)
    {
        ll f = data.first.first, s = data.first.second;
        string val = data.second;

        if(fix) f--, s--;
        out << f << " " << s << " " << val << '\n';
    }

    return 0;
}

int main(int argc, char *argv[])
{
    ios::sync_with_stdio(false);
	if(argc != 2)
	{
		cerr  << "./a.out [file to fix]" << endl;
		return 2;
	}
    solve(argv[1]);

    return 0;
}
