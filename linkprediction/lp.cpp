#include <vector>
#include <ctime>
#include <iostream>
#include <random>
#include <fstream>  
#include <string>  
#include <sstream>
#include <cmath>
#include <queue>
#include <io.h>

using namespace std;

typedef struct {
	int source;
	int dest;
	double weight;
	double score;
	bool isTest;
}link;

bool destLess(link l1,link l2) {
	return l1.dest < l2.dest;
}

bool scoreGreater(link l1, link l2) {
	return l1.score > l2.score;
}

class Network {
private:
	int nsize;//节点数量
	int M;//边数
	int testnum;//测试边数
	int begin;//开始节点
	int dmax;//最大的度
	double wmin;//最小权重
	double wmax;//最大权重
	int totd;
	vector<vector<link> > n;//邻接表
	vector<int> nlist;
	vector<link> linklist;//边集
	vector<link> templist;
	vector<vector<link> > tempn;
	vector<vector<link> > train;//训练集
	vector<link> test;//测试集
	vector<link> t;
	vector<link> nolink;//不存在边集
	vector<link> rank;
	vector<vector<double> > pset;
	vector<int> d;//
	vector<string> files;
	vector<vector<double> > eij;
	double AUC;
	queue<int> q; //搜索队列
	vector<bool> visited; //访问标志
	vector<int> level;//广度优先遍历层数
	default_random_engine e;//随机数引擎


public:

	void getAllFile(const char* path) {
		_finddata_t file;
		intptr_t lf;
		//输入文件夹路径
		if ((lf = _findfirst(path, &file)) == -1)  //若将*.*改为*.txt，则会输出该文件夹下所有的txt文件名
			cout << "Not Found!" << endl;
		else {
			//输出文件名
			files.push_back(file.name);
			while (_findnext(lf, &file) == 0) {
				files.push_back(file.name);
			}
		}
		_findclose(lf);
	}

	void readdata(string filename, bool isWeighted) {
		ifstream in(filename);
		string line;
		if (in) // 有该文件  
		{
			int n1, n2, temp;
			double w = 1;
			n.clear();
			linklist.clear();
			link l;
			nsize = 0;
			M = 0;
			begin = 0;
			wmin = 100;
			wmax = 0;
			dmax = 0;
			while (getline(in, line)) // line中不包括每行的换行符  
			{
				M++;
				istringstream is(line);
				if (isWeighted) {
					is >> n1 >> n2 >> w;
					wmin = w < wmin ? w : wmin;
					wmax = w > wmax ? w : wmax;
				}
				else {
					is >> n1 >> n2;
				}
				l.source = n1;
				l.dest = n2;
				l.weight = w;
				linklist.push_back(l);
				if ((n1 >= nsize) || (n2 >= nsize)) {
					nsize = ((n1 > n2) ? n1 : n2) + 1;
					n.resize(nsize);
				}
				n[n1].push_back(l);
				temp = l.source;
				l.source = l.dest;
				l.dest = temp;
				n[n2].push_back(l);
			}
			if (n[0].size() == 0) {
				begin = 1;
			}
			for (int i = 0; i < n.size(); i++) {
				sort(n[i].begin(), n[i].end(), destLess);
				dmax = (dmax > n[i].size()) ? dmax : n[i].size();
			}
			e.seed(time(0));
		}
		else // 没有该文件  
		{
			cout << "no such file" << endl;
			return;
		}
	}

	double countAssortative() {
		int d1, d2, dmin;
		double addai = 0, addei = 0;
		eij.clear();
		vector<double> ai(dmax + 1, 0);
		eij.resize(dmax + 1, ai);
		for (int i = 0; i < linklist.size(); i++) {
			d1 = n[linklist[i].source].size();
			d2 = n[linklist[i].dest].size();
			if (d1 != d2) {
				eij[d1][d2] += 0.5;
				eij[d2][d1] += 0.5;
			}
			else {
				eij[d1][d2]++;
			}		
		}
		for (int i = 1; i < eij.size(); i++) {
			addei += eij[i][i];
			for (int j = 1; j < eij[i].size(); j++) {
				ai[i] += eij[i][j];
			}
			addai += pow(ai[i], 2);
		}
		addei /= M;
		addai /= pow(M, 2);
		return (addei - addai) / (1 - addai);
	}

	void weightNormalize() {
		double dw = wmax - wmin;
		for (int i = 0; i < n.size(); i++) {
			for (int j = 0; j < n[i].size(); j++) {
				n[i][j].weight = (n[i][j].weight - wmin) / dw;
			}
		}
	}

	void init(double ratio) {
		testnum = M * ratio;
		train.clear();
		train.assign(n.begin(), n.end());
		test.clear();
		test.resize(testnum);
		templist.clear();
		templist.assign(linklist.begin(), linklist.end());
		dividedata();
		setnolink(testnum);
	}

	void countPset() {
		int totd;
		int dk;
		dmax = 0;
		int j;
		for (int i = 0; i < train.size(); i++) {
			dmax = (dmax > train[i].size()) ? dmax : train[i].size();
		}
		vector<double> pi(dmax + 1, 0);
		pset.clear();
		pset.resize(train.size(), pi);
		for (int i = 0; i < train.size(); i++) {
			totd = train[i].size();
			pset[i][0] = train[i].size();
			for (j = 0; j < train[i].size(); j++) {
				dk = train[train[i][j].dest].size();
				totd += dk;
				pset[i][j + 1] = dk;
			}
			for (int k = 0; k <= j; k++) {
				pset[i][k] /= totd;
			}
			sort(pset[i].begin(), pset[i].end(), greater<double>());
		}
	}

	double LRE(int n1, int n2) {
		double l = 0;
		int m = ((train[n1].size() < train[n2].size()) ? train[n1].size() : train[n2].size()) + 1;
		int d = distance(n1, n2);
		for (int i = 0; i < m; i++) {
			l += pset[n1][i] * log(pset[n1][i] / pset[n2][i]) + pset[n2][i] * log(pset[n2][i] / pset[n1][i]);//relative entropy 
		}
		l = l / d;
		return -l;
	}


	bool connected(int n1, int n2) {//用于验证删掉一条边后网络是否连通
		visited.clear();
		visited.resize(nsize, false);
		visited[n1] = true;
		queue<int> q;
		int v;
		int t = 0;
		for (int i = 0; i < train[n1].size(); i++) {
			if (train[n1][i].dest != n2) {
				q.push(train[n1][i].dest);
				visited[train[n1][i].dest] = true;
			}
		}
		while (!q.empty()) {
			t++;
			v = q.front();
			q.pop();
			for (int i = 0; i < train[v].size(); i++) {
				int now = train[v][i].dest;
				if (!visited[now]) {
					if (now == n2) {
						return true;
					}
					q.push(now);
					visited[now] = true;
				}
			}
			if (t > 5000) {//限制搜索次数，加快速度
				return false;
			}
		}
		return false;
	}

	double get_cluster() {
		double sum = 0;
		int ni;
		double ki;
		for (int i = 0; i < n.size(); i++) {
			ni = 0;
			ki = n[i].size();
			if (n[i].size() > 1) {
				for (int j = 0; j < n[i].size() - 1; j++) {
					for (int k = j + 1; k < n[i].size(); k++) {
						if (isNeighbor(n[i][j].dest, n[i][k].dest)) {
							ni++;
						}
					}
				}
				sum += 2 * ni / (ki*(ki - 1));
			}
		}
		return sum / n.size();
	}

	void dividedata() {
		int d;
		int n1, n2;
		link l;
		l.isTest = true;
		for (int i = 0; i < testnum;) {
			uniform_int_distribution<int> u1(0, templist.size() - 1);
			int pos = u1(e);
			l.source = n1 = templist[pos].source;
			l.dest = n2 = templist[pos].dest;
			if ((train[n1].size() > 1)&&((train[n2].size() > 1))&&(connected(n1, n2))) {
				templist.erase(templist.begin() + pos);
				test[i] = l;
				for (int j = 0; j < train[n1].size();j++) {
					if (train[n1][j].dest == n2) {
						train[n1].erase(train[n1].begin() + j);
					}
				}
				for (int k = 0; k < train[n2].size(); k++) {
					if (train[n2][k].dest == n1) {
						train[n2].erase(train[n2].begin() + k);
					}
				}
				i++;
			}
		}
	}

	/*void showInfo(const char* path, string result) {
		char path2[30];
		strcpy(path2, path);
		strcat(path2, "*.txt");
		getAllFile(path2);
		string fs = path;
		int n11, n12, n21, n22;
		double cn1, cn2, d1, d2;
		double p11, p12, p21, p22;
		for (int f = 0; f < files.size(); f++) {
			readdata(fs + files[f]);
			init(0.1);
			ofstream out(result+files[f]);
			for (int i = 0; i < test.size(); i++) {
				n11 = test[i][0];
				n12 = test[i][1];
				n21 = nolink[i][0];
				n22 = nolink[i][1];
				cn1 = CN(n11, n12);
				cn2 = CN(n21, n22);
				d1 = distance(n11, n12);
				d2 = distance(n21, n22);
				p11 = pn(n11, 1);
				p12 = pn(n12, 1);
				p21 = pn(n21, 1);
				p22 = pn(n22, 1);
				out << n11 << " " << n12 << " " << n21 << " " << n22 << " " << cn1 << " " << cn2 << " " << d1 << " " << d2 << " " << p11 << " " << p12 << " " << p21 << " " << p22 << " " << endl;
			}
			out.close();
		}
		
	}

	void outLinkInfo(string filename) {
		ofstream out(filename);
		int n1, n2;
		int d;
		int c;
		double l1;
		double l2;
		for (int i = 0; i < test.size(); i++) {
			n1 = test[i][0];
			n2 = test[i][1];
			d = distance(n1, n2);
			c = CN(n1, n2);
			l1 = le(n1);
			l2 = le(n2);
			out << n1 << "\t" << n2 << "\t" << d << "\t" << c << "\t" << l1 << "\t" << l2 << endl;
		}
	}

	void allLinkInfo(double ratio, const char* path, string outpath) {
		char path2[30];
		strcpy(path2, path);
		strcat(path2, "*.txt");
		getAllFile(path2);
		string fs = path;
		for (int f = 0; f < files.size(); f++) {
			readdata(fs + files[f]);
			init(ratio);
			outLinkInfo(outpath + files[f]);
		}
	}*/

	int CN(int n1, int n2) {
		int cn = 0;
		for (int i = 0; i < train[n1].size(); i++) {
			for (int j = 0; j < train[n2].size(); j++) {
				if (train[n1][i].dest == train[n2][j].dest) {
					cn++;
					break;
				}
			}
		}
		return cn;
	}

	double PE(int n1, int n2, int l) {
		double pe = 0;
		vector<vector<int> > path2;
		vector<vector<int> > path3;
		int p2i = 0, p3i = 0;
		for (int i = 0; i < train[n1].size(); i++) {
			for (int j = 0; j < train[n2].size(); j++) {
				if (train[n1][i].dest == train[n2][j].dest) {
					path2.resize(p2i + 1);
					path2[p2i].push_back(train[n1][i].dest);
					p2i++;
				}
			}
		}
		for (int i = 0; i < path2.size(); i++) {
			pe += IL1(n1, path2[i][0]) + IL1(path2[i][0], n2);
		}
		if (l == 2) {
			for (int i = 0; i < train[n1].size(); i++) {
				for (int j = 0; j < train[n2].size(); j++) {
					for (int k = 0; k < train[train[n1][i].dest].size(); k++) {
						if (train[train[n1][i].dest][k].dest == train[n2][j].dest) {
							path3.resize(p3i + 1);
							path3[p3i].push_back(train[n1][i].dest);
							path3[p3i].push_back(train[n2][j].dest);
							p3i++;
							break;
						}
					}
				}
			}
			for (int i = 0; i < path3.size(); i++) {
				pe += 0.5*(IL1(n1, path3[i][0]) + IL1(path3[i][0], path3[i][1]) + IL1(path3[i][1], n2));
			}
		}						
		return pe - IL1(n1, n2);
	}

	double AA(int n1, int n2) {
		double aa = 0;
		for (int i = 0; i < train[n1].size(); i++) {
			for (int j = 0; j < train[n2].size(); j++) {
				if (train[n1][i].dest == train[n2][j].dest) {
					aa += 1 / log(train[train[n1][i].dest].size());
					break;
				}
			}
		}
		return aa;
	}

	double RA(int n1, int n2) {
		double ra = 0;
		for (int i = 0; i < train[n1].size(); i++) {
			for (int j = 0; j < train[n2].size(); j++) {
				if (train[n1][i].dest == train[n2][j].dest) {
					ra += 1 / (double)train[train[n1][i].dest].size();
					break;
				}
			}
		}
		return ra;
	}

	void clear(queue<int>& q) {
		queue<int> empty;
		swap(empty, q);
	}

	double pn(int node, int maxlevel) {
		double addk = 0;
		int neighbor;//邻居
		clear(q);
		q.push(node);
		visited.clear();
		visited.resize(nsize, false);
		visited[node] = true;
		level.clear();
		level.resize(nsize, 0);
		while (!q.empty())
		{
			int v = q.front(); //取出队头的节点
			addk += train[v].size();
			q.pop();
			for (int i = 0; i < train[v].size(); i++) {
				neighbor = train[v][i].dest;
				if (!visited[neighbor]) {
					level[neighbor] = level[v] + 1;
					if (level[neighbor] <= maxlevel) {
						q.push(neighbor);
						visited[neighbor] = true;
					}
				}
			}
		}
		addk -= train[node].size();
		return (double)train[node].size() / addk;
	}

	double wpn(int node, int maxlevel) {
		double addw = 0;
		double nodew;
		int neighbor;
		clear(q);
		q.push(node);
		visited.clear();
		visited.resize(nsize, false);
		visited[node] = true;
		level.clear();
		level.resize(nsize, 0);
		while (!q.empty()) {
			int v = q.front();
			addw += allw(v);
			q.pop();
			for (int i = 0; i < train[v].size(); i++) {
				neighbor = train[v][i].dest;
				if (!visited[neighbor]) {
					level[neighbor] = level[v] + 1;
					if (level[neighbor] <= maxlevel) {
						q.push(neighbor);
						visited[neighbor] = true;
					}
				}
			}
		}
		addw -= allw(node);
		return allw(node) / addw;
	}

	double wwpn(int node, int maxlevel, double alpha) {
		double addw = 0;
		double nodew;
		int neighbor;
		clear(q);
		q.push(node);
		visited.clear();
		visited.resize(nsize, false);
		visited[node] = true;
		level.clear();
		level.resize(nsize, 0);
		while (!q.empty()) {
			int v = q.front();
			addw += wallw(v, alpha);
			q.pop();
			for (int i = 0; i < train[v].size(); i++) {
				neighbor = train[v][i].dest;
				if (!visited[neighbor]) {
					level[neighbor] = level[v] + 1;
					if (level[neighbor] <= maxlevel) {
						q.push(neighbor);
						visited[neighbor] = true;
					}
				}
			}
		}
		addw -= wallw(node, alpha);
		return wallw(node, alpha) / addw;
	}

	double wallw(int node, double alpha) {
		double alw = 0;
		for (int i = 0; i < train[node].size(); i++) {
			alw += pow(train[node][i].weight, alpha);
		}
		return alw;
	}

	double allw(int node) {
		double alw = 0;
		for (int i = 0; i < train[node].size(); i++) {
			alw += train[node][i].weight;
		}
		return alw;
	}

	void  countK() {
		int k;
		d.clear();
		totd = 0;
		for (int i = 0; i < train.size(); i++) {
			k = train[i].size();
			if (d.size() <= k) {
				d.resize(k + 1);
			}
			d[k]++;
		}
	}

	int distance(int n1, int n2) {
		int distance = 0;
		clear(q);
		q.push(n1);
		visited.clear();
		visited.resize(nsize, false);
		level.clear();
		level.resize(nsize, 0);
		visited[n1] = true;
		while (!q.empty())
		{
			int v = q.front(); //取出队头的节点
			q.pop();
			for (int i = 0; i < train[v].size(); i++) {
				int now = train[v][i].dest;
				if (!visited[now]) {
					level[now] = level[v] + 1;
					if (now == n2) {
						return level[now];
					}
					else {
						q.push(now);
						visited[now] = true;
					}
				}
			}
		}
		return 100;
	}
	
	double newlp(int n1, int n2, int l) {
		if((train[n1].size()==0)||(train[n2].size()==0)){
			return 0;
		}
		double p1 = pn(n1, l);
		double p2 = pn(n2, l);
		int d = distance(n1, n2);
		return (CN(n1, n2) + 1)*(-p1 * log(p1) - p2 * log(p2)) / (d - 1);
	}

	double wlp(int n1, int n2, int l) {
		if ((train[n1].size() == 0) || (train[n2].size() == 0)) {
			return 0;
		}
		double wp1 = wpn(n1, l);
		double wp2 = wpn(n2, l);
		int d = distance(n1, n2);
		return (-wp1 * log(wp1) - wp2 * log(wp2)) / (d - 1);
	}

	double wwlp(int n1, int n2, int l, double alpha) {
		if ((train[n1].size() == 0) || (train[n2].size() == 0)) {
			return 0;
		}
		double wwp1 = wwpn(n1, l, alpha);
		double wwp2 = wwpn(n2, l, alpha);
		int d = distance(n1, n2);
		return (-wwp1 * log(wwp1) - wwp2 * log(wwp2)) / (d - 1);
	}

	double wwlp2(int n1, int n2, int l, double alpha) {
		if ((train[n1].size() == 0) || (train[n2].size() == 0)) {
			return 0;
		}
		double wwp11 = wwpn(n1, l, alpha);
		double wwp21 = wwpn(n2, l, alpha);
		double wwp12 = pn(n1, l);
		double wwp22 = pn(n2, l);
		double wwp1 = wwp11 + wwp12;
		double wwp2 = wwp21 + wwp22;
		int d = distance(n1, n2);
		return (-wwp1 * log(wwp1) - wwp2 * log(wwp2)) / (d - 1);
	}

	double wwlp3(int n1, int n2, int l, double alpha) {
		if ((train[n1].size() == 0) || (train[n2].size() == 0)) {
			return 0;
		}
		double wwp1 = wwpn(n1, l, alpha);
		double wwp2 = wwpn(n2, l, alpha);
		double add = wwp1 + wwp2;
		wwp1 = wwp1 / add;
		wwp2 = wwp2 / add;
		int d = distance(n1, n2);
		return (-wwp1 * log(wwp1) - wwp2 * log(wwp2)) / (d - 1);
	}

	double hy(int n1, int n2, double a) {
		return a * newlp(n1, n2, 3) + (1 - a)*PE(n1, n2, 2);
	}

	double hy2(int n1, int n2) {
		return newlp(n1, n2, 3) * PE(n1, n2, 2);
	}

	double newcn(int n1, int n2) {
		if ((train[n1].size() == 0) || (train[n2].size() == 0)) {
			return 0;
		}
		double p1 = pn(n1, 1);
		double p2 = pn(n2, 1);
		int d = distance(n1, n2);
		return ((-p1 * log(p1) - p2 * log(p2)) / (d - 1)) + CN(n1, n2);
	}

	double lp5(int n1, int n2) {
		double l1 = le(n1);
		double l2 = le(n2);
		int d = distance(n1, n2);
		return (l1 + l2) / (d - 1);
	}
	
	double le(int node) {
		double pi;
		double l = 0;
		int totd = 0;
		for (int i = 0; i < train[node].size(); i++) {
			totd += train[train[node][i].dest].size();
		}
		for (int j = 0; j < train[node].size(); j++) {
			pi = (double)train[train[node][j].dest].size() / totd;
			l += pi * log2(pi);
		}
		if (l == 0) {
			system("pause");
		}
		return -l;
	}

	double sp(int n1, int n2) {
		return (double)1 / distance(n1, n2);
	}

	double LRW(int n1, int n2) {

	}

	double WCN(int n1, int n2) {
		double wcn = 0;
		for (int i = 0; i < train[n1].size(); i++) {
			for (int j = 0; j < train[n2].size(); j++) {
				if (train[n1][i].dest == train[n2][j].dest) {
					wcn+=train[n1][i].weight+train[n2][j].weight;
					break;
				}
			}
		}
		return wcn;
	}

	double WAA(int n1, int n2) {
		double waa = 0;
		double wcn;
		double sz;
		int z;
		for (int i = 0; i < train[n1].size(); i++) {
			for (int j = 0; j < train[n2].size(); j++) {
				if (train[n1][i].dest == train[n2][j].dest) {
					z = train[n1][i].dest;
					wcn = train[n1][i].weight + train[n2][j].weight;
					sz = 0;
					for (int k = 0; k < train[z].size(); k++) {
						sz += train[z][k].weight;
					}
					waa += wcn / log(1 + sz);
					break;
				}
			}
		}
		return waa;
	}

	double WRA(int n1, int n2) {
		double wra = 0;
		double wcn;
		double sz;
		int z;
		for (int i = 0; i < train[n1].size(); i++) {
			for (int j = 0; j < train[n2].size(); j++) {
				if (train[n1][i].dest == train[n2][j].dest) {
					z = train[n1][i].dest;
					wcn = train[n1][i].weight + train[n2][j].weight;
					sz = 0;
					for (int k = 0; k < train[z].size(); k++) {
						sz += train[z][k].weight;
					}
					wra += wcn / sz;
					break;
				}
			}
		}
		return wra;
	}

	double WWCN(int n1, int n2, double alpha) {
		double wwcn = 0;
		for (int i = 0; i < train[n1].size(); i++) {
			for (int j = 0; j < train[n2].size(); j++) {
				if (train[n1][i].dest == train[n2][j].dest) {
					wwcn += pow(train[n1][i].weight, alpha) + pow(train[n2][j].weight, alpha);
					break;
				}
			}
		}
		return wwcn;
	}

	double WWAA(int n1, int n2, double alpha) {
		double wwaa = 0;
		double wwcn;
		double sz;
		int z;
		for (int i = 0; i < train[n1].size(); i++) {
			for (int j = 0; j < train[n2].size(); j++) {
				if (train[n1][i].dest == train[n2][j].dest) {
					z = train[n1][i].dest;
					wwcn = pow(train[n1][i].weight, alpha) + pow(train[n2][j].weight, alpha);
					sz = 0;
					for (int k = 0; k < train[z].size(); k++) {
						sz += train[z][k].weight;
					}
					wwaa += wwcn / log(1 + sz);
					break;
				}
			}
		}
		return wwaa;
	}

	double WWRA(int n1, int n2, double alpha) {
		double wwra = 0;
		double wwcn;
		double sz;
		int z;
		for (int i = 0; i < train[n1].size(); i++) {
			for (int j = 0; j < train[n2].size(); j++) {
				if (train[n1][i].dest == train[n2][j].dest) {
					z = train[n1][i].dest;
					wwcn = pow(train[n1][i].weight, alpha) + pow(train[n2][j].weight, alpha);
					sz = 0;
					for (int k = 0; k < train[z].size(); k++) {
						sz += train[z][k].weight;
					}
					wwra += wwcn / sz;
					break;
				}
			}
		}
		return wwra;
	}

	double HEI(int n1, int n2, double alpha) {
		if (train[n1].size() == train[n2].size()) {
			return 0;
		}
		else {
			return pow(abs(int(train[n1].size() - train[n2].size())), alpha);
		}	
	}

	double HOI(int n1, int n2, double alpha) {
		return double(1) / pow(abs(int(train[n1].size() - train[n2].size())), alpha);
	}

	double HAI(int n1, int n2, double alpha) {
		double dk = abs(int(train[n1].size() - train[n2].size()));
		return alpha * dk + (1 - alpha)*dk;
	}

	double countAUC(int num, int method, double alpha) {//method:1.cn;2.pe
		int n11, n12;//测试边节点
		int n21, n22;//不存在边节点
		uniform_int_distribution<int> v1(0, test.size() - 1);
		uniform_int_distribution<int> v2(begin, nsize - 1);
		double s1, s2;
		double t = 0;//测试边相似度比不存在边大
		for (int i = 0; i < num; i++) {
			//随机选择测试边和不存在边各一条，分别计算相似度比较
			int testn = v1(e);
			n11 = test[testn].source;
			n12 = test[testn].dest;
			do
			{
				n21 = v2(e);
				n22 = v2(e);
			} while ((n21 == n22) || isNeighbor(n21, n22) || (train[n21].size() == 0) || (train[n22].size() == 0));
			s1 = sim(n11, n12, method, alpha);//测试边
			s2 = sim(n21, n22, method, alpha);//不存在边	
 			if (s1 > s2) {
				t++;
			}
			else if (s1 == s2) {
				t += 0.5;
			}
		}
		AUC = t / num;
		return AUC;
	}

	double AUC2(int num, int method, double alpha) {
		int n11, n12;//测试边节点
		int n21, n22;//不存在边节点
		int tn, nn;
		setnolink(test.size());
		uniform_int_distribution<int> v1(0, test.size() - 1);
		uniform_int_distribution<int> v2(0, nolink.size() - 1);
		double s1, s2;
		double t = 0;//测试边相似度比不存在边大
		for (int i = 0; i < num; i++) {
			//随机选择测试边和不存在边各一条，分别计算相似度比较
			tn = v1(e);
			nn = v2(e);
			n11 = test[tn].source;
			n12 = test[tn].dest;
			n21 = nolink[nn].source;
			n22 = nolink[nn].dest;
			s1 = sim(n11, n12, method, alpha);//测试边
			s2 = sim(n21, n22, method, alpha);//不存在边			
			if (s1 > s2) {
				t++;
			}
			else if (s1 == s2) {
				t += 0.5;
			}
		}
		AUC = t / num;
		return AUC;
	}

	void setnolink(int size) {
		int n1, n2, temp;
		link l;
		l.isTest = false;
		nolink.clear();
		nolink.resize(size);
		tempn.clear();
		tempn.assign(n.begin(), n.end());
		uniform_int_distribution<int> v(begin, nsize - 1);
		for (int i = 0; i < size; i++) {
			do
			{
				n1 = v(e);
				n2 = v(e);
			} while ((train[n1].size() == 0) || (train[n2].size() == 0) || (n1 == n2) || isNeighbor2(n1, n2));
			l.source = n1;
			l.dest = n2;
			tempn[n1].push_back(l);
			nolink[i] = l;
			temp = l.source;
			l.source = l.dest;
			l.dest = temp;
			tempn[n2].push_back(l);
		}
	}

	void setalllink() {
		int n1, n2;
		link l;
		int del;
		l.isTest = false;
		nolink.clear();
		for (int i = 0; i < n.size(); i++) {
			if (n[i].size() > 0) {
				del = 0;
				while ((n[i][del].dest < (i + 1)) && (del < (n[i].size() - 1))) {
					del++;
				}
				for (int j = i + 1; j < n.size(); j++) {
					if (j == n[i][del].dest) {
						if (del < (n[i].size() - 1)) {
							del++;
						}
						continue;
					}
					l.source = i;
					l.dest = j;
					nolink.push_back(l);
				}
			}
		}
	}

	double Precision(int method, int L, double alpha) {
		int lm = 0;
		//L = L < testnum ? L : testnum;
		setalllink();
		rank.clear();
		rank.insert(rank.end(), test.begin(), test.end());
		rank.insert(rank.end(), nolink.begin(), nolink.end());
		for (int i = 0; i < rank.size(); i++) {
			rank[i].score = sim(rank[i].source, rank[i].dest, method, alpha);
		}

		sort(rank.begin(), rank.end(), scoreGreater);
		for (int i = 0; i < L; i++) {
			if (rank[i].isTest) {
				lm++;
			}
		}
		return (double)lm / L;
	}

	double sim(int n1, int n2, int method, double alpha) {
		switch (method)
		{
		case 0:
			return hy(n1, n2, alpha);
			break;
		case 1:
			return CN(n1, n2);
			break;
		case 2:
			return AA(n1, n2);
			break;
		case 3:
			return RA(n1, n2);
			break;
		case 4:
			return PE(n1, n2, 1);
			break;
		case 5:
			return newlp(n1, n2, 1);
			break;
		case 6:
			return hy(n1, n2, 0.9);
			break;
		case 7:
			return hy2(n1, n2);
			break;
		case 8:
			return sp(n1, n2);
			break;
		case 9:
			return LRE(n1, n2);
			break;
		case 10:
			return WWCN(n1, n2, alpha);
			break;
		case 11:
			return WWAA(n1, n2, alpha);
			break;
		case 12:
			return WWRA(n1, n2, alpha);
			break;
		case 13:
			return wwlp(n1, n2, 1, alpha);
			break;
		case 14:
			return wwlp2(n1, n2, 1, alpha);
			break;
		case 15:
			return wwlp3(n1, n2, 1, alpha);
			break;
		case 16:
			return HEI(n1, n2, alpha);
			break;
		case 17:
			return HAI(n1, n2, alpha);
			break;
		}
	}

	bool isNeighbor(int n1, int n2) {
		for (int i = 0; i < n[n1].size(); i++) {
			if (n[n1][i].dest == n2) {
				return true;
			}
		}
		for (int j = 0; j < n[n2].size(); j++) {
			if (n[n2][j].dest == n1) {
				return true;
			}
		}
		return false;
	}

	bool isNeighbor2(int n1, int n2) {
		for (int i = 0; i < tempn[n1].size(); i++) {
			if (tempn[n1][i].dest == n2) {
				return true;
			}
		}
		return false;
	}

	double IL1(int na, int nb) {
		int ka = train[na].size();
		int kb = train[nb].size();
		double c = 1;
		for (int i = 0; i <= kb - 1; i++) {
			c *= (double)(M - ka - i) / (M - i);
		}
		return -log2(1 - c);
	}


	void tries(int times, int AUCtimes, int isAUC, double ratio, const char* path, string resultfile, bool isWeighted) {
		char path2[30];
		strcpy(path2, path);
		strcat(path2, "*.txt");
		getAllFile(path2);
		string fs = path;
		double result = 0;
		clock_t start, end;
		double ai;
		vector<int> method = { 1,4,5 };
		ofstream out(resultfile);
		for (int f = 0; f < files.size(); f++) {
			readdata(fs + files[f], isWeighted);
			out << files[f] << " ";
			cout << files[f] << endl;
			for (int m = 0; m < method.size(); m++) {
				cout <<"method: "<< method[m] << endl;
				start = clock();
				result = 0;
				for (int t = 1; t <= times; t++) {
					cout << t << ":" << endl;
					init(ratio);
					if (method[m] == 9) {
						countPset();
					}
					switch (isAUC)
					{
					case 1:
						ai =countAUC(AUCtimes, method[m], 0);
						break;
					case 0:
						ai = Precision(method[m], 100, 0);
					}
					result += ai;
					cout << ai <<endl;
				}
				result = result / times;
				cout << result << endl;
				out << result << " ";
				end = clock();
				cout << end - start << "ms" << endl;
			}
			out << endl;
		}	
		out.close();
	}

	void triesalpha(int times, int isAUC, double ratio, const char* path, string resultfile) {
		double a;
		char path2[30];
		strcpy(path2, path);
		strcat(path2, "*.txt");
		getAllFile(path2);
		string fs = path;
		double result = 0;
		double ai;
		vector<int> method = { 16 };
		ofstream out(resultfile);
		for (int f = 0; f < files.size(); f++) {//遍历每个网络
			readdata(fs + files[f], true);
			out << files[f] << endl;
			cout << files[f] << endl;
			for (int m = 0; m < method.size(); m++) {//遍历每种算法
				cout << "method: " << method[m] << endl;
				a = -1;
				while (a < 1.1) {//遍历alpha的取值
					result = 0;
					for (int t = 1; t <= times; t++) {
						init(ratio);
						switch (isAUC)
						{
						case 1:
							ai = countAUC(10000, method[m], a);
							out << ai << " ";
							break;
						case 0:
							ai = Precision(method[m], 100, a);
						}
						result += ai;
					}
					out << endl;
					cout << a << endl;
					result = result / times;
					cout << result << endl;
					//out << result << endl;
					a += 0.1;
				}
				out << endl;
			}
			out << endl;
		}
		out.close();
	}

	void triesbeta(int times, int isAUC, double ratio, const char* path, string resultfile) {
		double b;
		char path2[30];
		strcpy(path2, path);
		strcat(path2, "*.txt");
		getAllFile(path2);
		string fs = path;
		double result = 0;
		double ai;
		ofstream out(resultfile);
		for (int f = 0; f < files.size(); f++) {//遍历每个网络
			readdata(fs + files[f], true);
			out << files[f] << endl;
			cout << files[f] << endl;
			b = 0;
			while (b < 1.1) {//遍历alpha的取值
				result = 0;
				for (int t = 1; t <= times; t++) {
					init(ratio);
					switch (isAUC)
					{
					case 1:
						ai = countAUC(1000, 0, b);
						break;
					case 0:
						ai = Precision(0, 100, b);
					}
					result += ai;
				}
				cout << b << endl;
				result = result / times;
				cout << result << endl;
				out << result << " ";
				b += 0.1;
			}
			out << endl;
		}
		out.close();
	}

	void showInfo(string infile, string outfile) {
		int n11, n12, n21, n22;
		readdata(infile, false);
		init(0.1);
		ofstream out(outfile);
		for (int i = 0; i < test.size(); i++) {
			n11 = test[i].source;
			n12 = test[i].dest;
			n21 = nolink[i].source;
			n22 = nolink[i].dest;
			out << n11 << " " << n12 << " " << n21 << " " << n22 << " " << pn(n11, 1) << " " << pn(n12, 1) << " " << distance(n11, n12) << " " << pn(n21, 1) << " " << pn(n22, 1) << " " << distance(n21, n22) << " " << CN(n11, n12) << " " << CN(n21, n22) << endl;
		}
	}

	/*void isDirected(string filename) {
		readdata(filename, true, false);
		cout << filename << " ";
		for (int i = 0; i < n.size(); i++) {
			if (n[i].size() > 1) {
				for (int j = 0; j < n[i].size() - 1; j++) {
					for (int k = j + 1; k < n[i].size(); k++) {
						if (n[i][j].dest == n[i][k].dest) {
							cout << n[i][j].dest << " " << i << endl;
							cout << "Directed" << endl;
							return;
						}
					}
				}
			}
		}
		cout << "Undirected" << endl;
	}

	void testDirected() {
		getAllFile("F:/data/lp_data/*.txt");
		for (int f = 0; f < files.size(); f++) {
			isDirected(files[f]);
		}
	}*/

	void makeUndirected(string infile, string outfile, bool isWeighted) {
		ifstream in(infile);
		string line;
		if (in) // 有该文件  
		{
			int n1, n2, temp;
			double w = 1;
			n.clear();
			linklist.clear();
			nsize = 0;
			M = 0;
			begin = 0;
			link l;
			while (getline(in, line)) // line中不包括每行的换行符  
			{
				M++;
				istringstream is(line);
				if (isWeighted) {
					is >> n1 >> n2 >> w;
				}
				else {
					is >> n1 >> n2;
				}
				if ((n1 >= nsize) || (n2 >= nsize)) {
					nsize = ((n1 > n2) ? n1 : n2) + 1;
					n.resize(nsize);
				}
				if (n1 != n2) {
					if (isNeighbor(n1, n2)) {//边已存在
						for (int i = 0; i < n[n1].size(); i++) {
							if (n[n1][i].dest == n2) {
								n[n1][i].weight += w;
							}
						}
						for (int j = 0; j < n[n2].size(); j++) {
							if (n[n2][j].dest == n1) {
								n[n2][j].weight += w;
							}
						}
					}
					else {//边不存在
						l.source = n1;
						l.dest = n2;
						l.weight = w;
						n[n1].push_back(l);
						temp = l.source;
						l.source = l.dest;
						l.dest = temp;
						n[n2].push_back(l);
					}
				}				
			}
			for (int i = 0; i < n.size(); i++) {
				sort(n[i].begin(), n[i].end(), destLess);
			}
			in.close();
			ofstream out(outfile);
			for (int i = 0; i < n.size(); i++) {
				for (int j = 0; j < n[i].size(); j++) {
					if (n[i][j].dest > i) {
						out << i << " " << n[i][j].dest << " " << n[i][j].weight << endl;
					}
				}
			}
			out.close();
		}
		else // 没有该文件  
		{
			cout << "no such file" << endl;
			return;
		}
		cout << outfile << " loaded" << endl;
	}

	double countH() {
		double kave = 0, k2ave = 0;
		for (int i = 0; i < n.size(); i++) {
			kave += n[i].size();
			k2ave += pow(n[i].size(), 2);
		}
		return k2ave / nsize / pow(kave / nsize, 2);
	}

	/*void allUndirected() {
		string fs = "F:/data/lp_data/";
		getAllFile("F:/data/lp_data/*.txt");
		for (int f = 0; f < files.size(); f++) {
			makeUndirected(fs + files[f]);
		}
	}*/

	void dataProcess() {
		nlist.clear();
		int n1, n2;
		double w = 1;
		ifstream in1("F:/data/lp_data/retweet1.txt");
		ifstream in2("F:/data/lp_data/retweet2.txt");
		ofstream out("F:/data/lp_data/retweet_U.txt");
		string line;
		if (in1)
		{
			int ni;
			while (getline(in1, line))
			{
				istringstream is(line);
				is >> ni;
				nlist.push_back(ni);
			}
		}
		if (in2) {
			while (getline(in2, line))
			{
				istringstream is(line);
				is >> n1 >> n2 >> w;
				out << lower_bound(nlist.begin(), nlist.end(), n1) - nlist.begin() << " " << lower_bound(nlist.begin(), nlist.end(), n2) - nlist.begin() << " " << w << endl;
			}
		}
		in1.close();
		in2.close();
		out.close();
	}

	void nullmodel_0k(string infile, string outfile, bool isWeighted, int times) {
		readdata(infile, isWeighted);
		times *= nsize;
		int pos;
		int n1, n2;
		double w;
		uniform_int_distribution<int> u1(0, linklist.size() - 1);
		uniform_int_distribution<int> u2(0, nsize - 1);
		while (times > 0) {
			pos = u1(e);
			n1 = linklist[pos].source;
			n2 = linklist[pos].dest;
			w = linklist[pos].weight;
			if (connected2(n1, n2)) {
				nErase(n1, n2);
				do {
					n1 = u2(e);
					n2 = u2(e);
				} while ((n1 == n2) || (isNeighbor(n1, n2)));
				nAdd(n1, n2, w);
				linklist[pos].source = n1;
				linklist[pos].dest = n2;
				times--;
			}		
		}
		ofstream out(outfile);
		if (isWeighted) {
			for (int i = 0; i < linklist.size(); i++) {
				out << linklist[i].source << " " << linklist[i].dest << " " << linklist[i].weight << endl;
			}
		}
		else {
			for (int i = 0; i < linklist.size(); i++) {
				out << linklist[i].source << " " << linklist[i].dest << endl;
			}
		}
	}

	bool connected2(int n1, int n2) {//用于验证删掉一条边后网络是否连通
		visited.clear();
		visited.resize(nsize, false);
		visited[n1] = true;
		queue<int> q;
		int v;
		int t = 0;
		for (int i = 0; i < n[n1].size(); i++) {
			if (n[n1][i].dest != n2) {
				q.push(n[n1][i].dest);
				visited[n[n1][i].dest] = true;
			}
		}
		while (!q.empty()) {
			t++;
			v = q.front();
			q.pop();
			for (int i = 0; i < n[v].size(); i++) {
				int now = n[v][i].dest;
				if (!visited[now]) {
					if (now == n2) {
						return true;
					}
					q.push(now);
					visited[now] = true;
				}
			}
			if (t > 5000) {//限制搜索次数，加快速度
				return false;
			}
		}
		return false;
	}

	void nErase(int n1, int n2) {
		for (int i = 0; i < n[n1].size(); i++) {
			if (n[n1][i].dest == n2) {
				n[n1].erase(n[n1].begin() + i);
				break;
			}
		}
		for (int i = 0; i < n[n2].size(); i++) {
			if (n[n2][i].dest == n1) {
				n[n2].erase(n[n2].begin() + i);
				break;
			}
		}
	}

	void nAdd(int n1, int n2, double w) {
		link l;
		l.source = n1;
		l.dest = n2;
		l.weight = w;
		n[n1].push_back(l);
		l.source = n2;
		l.dest = n1;
		n[n2].push_back(l);
	}

	void getPart(int num, string outfile) {
		ofstream out(outfile);
		nlist.clear();
		clear(q);
		int n1, n2;
		uniform_int_distribution<int> u(0, nsize - 1);
		do {
			n1 = u(e);
		} while (n[n1].size() <= 0);
		q.push(n1);
		visited.clear();
		visited.resize(nsize, false);
		visited[n1] = true;
		while (num > 0)
		{
			int v = q.front(); //取出队头的节点
			q.pop();
			nlist.push_back(v);
			for (int i = 0; i < n[v].size(); i++) {
				int now = n[v][i].dest;
				if (!visited[now]) {
					q.push(now);
					visited[now] = true;
				}
			}
			num--;
		}
		sort(nlist.begin(), nlist.end());
		for (int i = 0; i < linklist.size(); i++) {
			n1 = linklist[i].source;
			n2 = linklist[i].dest;
			if (binary_search(nlist.begin(), nlist.end(), n1) && binary_search(nlist.begin(), nlist.end(), n2)) {
				out << n1 << " " << n2 << endl;
			}
		}
		out.close();
	}

};

int main(int argc, char **argv) {
	Network g;
	//g.readdata("F:/data/lp_data/CGScience.txt", false);
	//g.getPart(500, "F:/data/lp_data/CGScience_part1.txt");
	//g.tries(100, 10000, true, 0.1, "F:/data/temp/", "F:/data/lp_data/result/712_AUC.txt", false);
	//g.nullmodel_0k("F:/data/lp_data/football.txt", "F:/data/lp_data/football_0k.txt", false, 10);
	//g.dataProcess();
	//g.triesalpha(100, 1, 0.1, "F:/data/temp/", "F:/data/lp_data/result/712_AUC2.txt");
	//g.makeUndirected("F:/data/lp_data/weighted/U/twitter/retweet.txt", "F:/data/lp_data/weighted/U/twitter/retweet_U.txt", true);
	//g.readdata("F:/data/lp_data/football.txt", false);
	g.showInfo("F:/data/lp_data/CGScience.txt", "F:/data/lp_data/CGScience_info.txt");
	//cout << g.countAssortative() << endl;
	system("pause");
}