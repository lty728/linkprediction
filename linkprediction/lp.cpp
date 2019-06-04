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
	int totd;
	vector<vector<link> > n;//邻接表
	vector<link> linklist;//边集
	vector<link> templist;
	vector<vector<link> > tempn;
	vector<vector<link> > train;//训练集
	vector<link> test;//测试集
	vector<link> t;
	vector<link> nolink;//不存在边集
	vector<vector<double> > pset;
	vector<int> d;//
	vector<string> files;
	double AUC;
	double precision;
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

	void readdata(string filename, bool isWeighted, bool isDirected) {
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
				l.source = n1;
				l.dest = n2;
				l.weight = w;
				linklist.push_back(l);
				if ((n1 >= nsize) || (n2 >= nsize)) {
					nsize = ((n1 > n2) ? n1 : n2) + 1;
					n.resize(nsize);
				}
				n[n1].push_back(l);
				if (!isDirected) {
					temp = l.source;
					l.source = l.dest;
					l.dest = temp;
					n[n2].push_back(l);
				}
			}
			if (n[0].size() == 0) {
				begin = 1;
			}
			e.seed(time(0));
		}
		else // 没有该文件  
		{
			cout << "no such file" << endl;
			return;
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
		for (int i = 0; i < train[n1].size(); i++) {
			if (train[n1][i].dest != n2) {
				q.push(train[n1][i].dest);
				visited[train[n1][i].dest] = true;
			}
		}
		while (!q.empty()) {
			int v = q.front();
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
		for (int i = 0; i < testnum;) {
			uniform_int_distribution<int> u1(0, templist.size() - 1);
			int pos = u1(e);
			l.source = templist[pos].source;
			l.dest = templist[pos].dest;
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

	double PE(int n1, int n2) {
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
		for (int i = 0; i < path2.size(); i++) {
			pe += IL1(n1, path2[i][0]) + IL1(path2[i][0], n2);
		}
		for (int i = 0; i < path3.size(); i++) {
			pe += 0.5*(IL1(n1, path3[i][0]) + IL1(path3[i][0], path3[i][1]) + IL1(path3[i][1], n2));
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

	double hy(int n1, int n2, double a) {
		return a * newlp(n1, n2, 3) + (1 - a)*PE(n1, n2);
	}

	double hy2(int n1, int n2) {
		return newlp(n1, n2, 3) * PE(n1, n2);
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
					for (int k = 0; k < train[z].size; z++) {
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
					for (int k = 0; k < train[z].size; z++) {
						sz += train[z][k].weight;
					}
					wra += wcn / sz;
					break;
				}
			}
		}
		return wra;
	}

	double countAUC(int num, int method) {//method:1.cn;2.pe
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
			s1 = sim(n11, n12, method);//测试边
			s2 = sim(n21, n22, method);//不存在边			
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

	double AUC2(int num, int method) {
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
			s1 = sim(n11, n12, method);//测试边
			s2 = sim(n21, n22, method);//不存在边			
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

	double Precision(int method) {
		double p = 0;
		return p;
	}

	double sim(int n1, int n2, int method) {
		switch (method)
		{
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
			return PE(n1, n2);
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
		}
	}

	bool isNeighbor(int n1, int n2) {
		for (int i = 0; i < n[n1].size(); i++) {
			if (n[n1][i].dest == n2) {
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


	void tries(int times, int AUCtimes, int isAUC, double ratio, const char* path, string resultfile, bool isWeighted, bool isDirected) {
		char path2[30];
		strcpy(path2, path);
		strcat(path2, "*.txt");
		getAllFile(path2);
		string fs = path;
		double result = 0;
		clock_t start, end;
		double ai;
		vector<int> method = { 7 };
		ofstream out(resultfile);
		for (int f = 0; f < files.size(); f++) {
			readdata(fs + files[f], isWeighted, isDirected);
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
						ai = countAUC(AUCtimes, method[m]);
						break;
					case 0:
						ai = Precision(method[m]);
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
	}

	/*void isDirected(string filename) {
		string fs = "F:/data/lp_data/";
		readdata(fs + filename);
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

	void makeUndirected(string filename) {
		ifstream in(filename);
		string line;
		if (in) // 有该文件  
		{
			int n1, n2, temp;
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
				is >> n1 >> n2;				
				if ((n1 >= nsize) || (n2 >= nsize)) {
					nsize = ((n1 > n2) ? n1 : n2) + 1;
					n.resize(nsize);
				}
				if ((n1 != n2) && !isNeighbor(n1, n2)) {
					l.source = n1;
					l.dest = n2;
					linklist.push_back(l);
					n[n1].push_back(l);
					temp = l.source;
					l.source = l.dest;
					l.dest = temp;
					n[n2].push_back(l);
				}				
			}
			in.close();
			ofstream out(filename);
			for (int i = 0; i < linklist.size(); i++) {
				out << linklist[i].source << " " << linklist[i].dest << endl;
			}
			out.close();
		}
		else // 没有该文件  
		{
			cout << "no such file" << endl;
			return;
		}
		cout << filename << " loaded" << endl;
	}

	void allUndirected() {
		string fs = "F:/data/lp_data/";
		getAllFile("F:/data/lp_data/*.txt");
		for (int f = 0; f < files.size(); f++) {
			makeUndirected(fs + files[f]);
		}
	}

};

int main(int argc, char **argv) {
	Network g;
	g.readdata("F:/data/lp_data/weighted/moreno_train", true, false);
	//g.allLinkInfo(0.1, "F:/data/lp_data/test/", "F:/data/lp_data/info/");
	//g.testDirected();
	//g.allUndirected();
	//g.tries(10, 1000, 1, 0.1, "F:/data/lp_data/test/", "F:/data/lp_data/result/result_hy2.txt");
	//g.showInfo("F:/data/lp_data/test/", "F:/data/lp_data/result/info/");
	//g.tries(100, 1000, 0, 0.1, "F:/data/lp_data/", "F:/data/lp_data/result/precision.txt");
	//g.init(0.1);8
	//cout << g.get_cluster();
	system("pause");
}