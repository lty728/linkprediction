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
	int dis;
	int cn;
	double score;
	bool isTest;
	vector<int> path_num;
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
	vector<string> files;
	vector<vector<double> > eij;
	vector<int> path;//存放当前路径
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
			double w = 0;
			n.clear();
			linklist.clear();
			link l;
			nsize = 0;
			M = 0;
			begin = 0;
			dmax = 0;
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

	void init(double ratio) {
		testnum = M * ratio;
		train.clear();
		train.assign(n.begin(), n.end());
		test.clear();
		test.resize(testnum);
		templist.clear();
		templist.assign(linklist.begin(), linklist.end());
		dividedata(true);
		setnolink(testnum);
	}

	void clear(queue<int>& q) {
		queue<int> empty;
		swap(empty, q);
	}

	void all_paths(link &l, int cutoff) {//所有长度不大于cutoff的路径
		l.path_num.clear();
		l.path_num.resize(cutoff + 1, 0);
		visited.clear();
		visited.resize(nsize, false);
		path.clear();
		dfs(l, l.source, l.dest, path, cutoff);
	}

	/*void all_sp(link &l,int cutoff) {//所有最短路径
		l.sp_num = 0;
		visited.clear();
		visited.resize(nsize, false);
		path.clear();
		dfs_sp(l, l.source, l.dest, path, cutoff);
	}*/

	void dfs(link &l, int s, int d, vector<int> &path, int cutoff) {
		visited[s] = true;
		path.push_back(s);
		if (s == d) {
			l.path_num[path.size() - 1]++;
		}
		else {
			if ((path.size() - 1) >= cutoff) {
				path.pop_back();
				visited[s] = false;
				return;
			}
			else {
				for (int i = 0; i < train[s].size(); i++) {
					if (!visited[train[s][i].dest]) {
						dfs(l,train[s][i].dest, d, path, cutoff);
					}
				}
			}
		}
		path.pop_back();
		visited[s] = false;
	}

	/*void dfs_sp(link &l, int s, int d, vector<int> &path, int cutoff) {
		visited[s] = true;
		path.push_back(s);
		if (s == d) {
			l.sp_num++;
		}
		else {
			if ((path.size() - 1) >= cutoff) {
				path.pop_back();
				visited[s] = false;
				return;
			}
			else {
				for (int i = 0; i < train[s].size(); i++) {
					if (!visited[train[s][i].dest]) {
						dfs_sp(l, train[s][i].dest, d, path, cutoff);
					}
				}
			}
		}
		path.pop_back();
		visited[s] = false;
	}*/

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
			/*if (t > 10000) {//限制搜索次数，加快速度
				return false;
			}*/
		}
		return false;
	}

	void dividedata(bool isConnected) {
		int d;
		int n1, n2;
		link l;
		l.dis = 0;
		l.cn = -1;
		l.isTest = true;
		for (int i = 0; i < testnum;) {
			uniform_int_distribution<int> u1(0, templist.size() - 1);
			int pos = u1(e);
			l.source = n1 = templist[pos].source;
			l.dest = n2 = templist[pos].dest;
			if (isConnected) {
				if ((train[n1].size() > 1) && ((train[n2].size() > 1)) && (connected(n1, n2))) {
					templist.erase(templist.begin() + pos);
					test[i] = l;
					for (int j = 0; j < train[n1].size(); j++) {
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
			else
			{
				templist.erase(templist.begin() + pos);
				test[i] = l;
				for (int j = 0; j < train[n1].size(); j++) {
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

	int CN(link &l) {
		if (l.cn < 0) {
			int n1, n2;
			n1 = l.source;
			n2 = l.dest;
			l.cn = 0;
			for (int i = 0; i < train[n1].size(); i++) {
				for (int j = 0; j < train[n2].size(); j++) {
					if (train[n1][i].dest == train[n2][j].dest) {
						l.cn++;
						break;
					}
				}
			}
		}	
		return l.cn;
	}

	int CN2(int n1,int n2) {
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

	double PE(int n1, int n2, int l) {
		if (train[n1].size()*train[n2].size() == 0) {
			return 0;
		}
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

	double FL(link &l, int cutoff) {
		double fl = 0;
		int div;
		if (l.path_num.size() == 0) {
			all_paths(l, cutoff);
		}
		for (int i = 2; i <= cutoff; i++) {
			div = 1;
			for (int j = 2; j <= i; j++) {
				div *= nsize - j;
			}
			fl += (double)l.path_num[i] / ((i - 1)*div);
		}
		return fl;
	}

	double LP(link &l, double alpha) {
		if (l.path_num.size() <= 3) {
			all_paths(l, 3);
		}
		return (double)l.path_num[2] + alpha * l.path_num[3];
	}

	double pn(int node, int maxlevel) {
		if (train[node].size() == 0) {
			return 0;
		}
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
		return 1000;
	}

	double newlp(link &l, int le) {
		if ((train[l.source].size() == 0) || (train[l.dest].size() == 0)) {
			return 0;
		}
		if (l.dis == 0) {
			l.dis = distance(l.source, l.dest);
		}
		if (l.dis <= 1) {
			return 0;
		}
		double p1 = pn(l.source, le);
		double p2 = pn(l.dest, le);
		p1 = p1 / (p1 + p2);
		p2 = 1 - p1;
		return (-p1 * log(p1) - p2 * log(p2)) / (l.dis - 1);
	}

	double hy(link &l, double alpha) {
		if (l.dis == 0) {
			l.dis = distance(l.source, l.dest);
		}
		if (l.cn < 0) {
			CN(l);
		}
		return (double)1 / (l.dis - 1) + alpha * l.cn;
	}

	double hy2(link &l) {
		if (l.dis == 0) {
			l.dis = distance(l.source, l.dest);
		}
		if (l.cn < 0) {
			CN(l);
		}
		return (double)1 / (l.dis - 1) + l.cn;
	}

	double sp(link &l) {
		if (l.dis == 0) {
			l.dis = distance(l.source, l.dest);
		}
		return (double)1 / (l.dis - 1);
	}

	/*double countAUC(int num, int method, double alpha) {//method:1.cn;2.pe
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
	}*/

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
			nn = v2(e);;
			s1 = sim(test[tn], method, alpha);//测试边
			s2 = sim(nolink[nn], method, alpha);//不存在边			
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
		l.dis = 0;
		l.cn = -1;
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
		l.dis = 0;
		l.cn = -1;
		rank.clear();
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
					rank.push_back(l);
				}
			}
		}
	}

	double sim(link &l, int method, double alpha) {
		int n1 = l.source;
		int n2 = l.dest;
		switch (method)
		{
		case 1:
			return CN(l);
			break;
		case 2:
			return AA(n1, n2);
			break;
		case 3:
			return RA(n1, n2);
			break;
		case 4:
			return PE(n1, n2, 2);
			break;
		case 5:
			return sp(l);
			break;
		case 6:
			return LP(l, 0.01);
			break;
		case 7:
			return FL(l, 3);
			break;
		case 9:
			return hy2(l);
			break;
		case 10:
			return hy(l, 2);
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

	void triesAUC(int times, int AUCtimes, int isAUC, double ratio, const char* path, string resultfile, bool isWeighted) {
		char path2[30];
		strcpy(path2, path);
		strcat(path2, "*.txt");
		getAllFile(path2);
		string fs = path;
		double rm;
		int tn, nn;
		double s1, s2;
		vector<int> method = { 5,10 };
		vector<vector<vector<double> > > res(files.size(), vector<vector<double> >(method.size(), vector<double>(times, 0)));//结果列表
		ofstream out(resultfile);
		for (int f = 0; f < files.size(); f++) {
			readdata(fs + files[f], isWeighted);
			cout << files[f] << endl;
			for (int t = 0; t < times; t++) {
				cout << t << " ";
				init(ratio);
				setnolink(test.size());
				cout << "divided" << endl;
				uniform_int_distribution<int> v1(0, test.size() - 1);
				uniform_int_distribution<int> v2(0, nolink.size() - 1);
				for (int at = 0; at < AUCtimes; at++) {//
					tn = v1(e);
					nn = v2(e);
					for (int m = 0; m < method.size(); m++) {
						s1 = sim(test[tn], method[m], 0);//测试边
						s2 = sim(nolink[nn], method[m], 0);
						if (s1 > s2) {
							res[f][m][t] += 1;
						}
						else if (s1 == s2) {
							res[f][m][t] += 0.5;
						}
					}
				}
			}
		}
		for (int m = 0; m < method.size(); m++) {
			out << " " << method[m];
		}
		out << endl;
		for (int f = 0; f < files.size(); f++) {
			out << files[f];
			for (int m = 0; m < method.size(); m++) {
				rm = 0;
				for (int t = 0; t < times; t++) {
					rm += res[f][m][t] / AUCtimes;
				}
				out << " " << rm / times;
			}
			out << endl;
		}
		out.close();
	}

	void triesprecision(int times, double top_ratio, double ratio, const char* path, string resultfile, bool isWeighted) {
		char path2[30];
		strcpy(path2, path);
		strcat(path2, "*.txt");
		getAllFile(path2);
		string fs = path;
		vector<int> method = { 1,2,3,4,6,10,11,12,13,14,15,16,17,18,19,20 };
		//vector<int> method = { 10 };
		double rt = 0;
		int topL = 100;
		vector<vector<vector<double> > > res(files.size(), vector<vector<double> >(method.size(), vector<double>(times, 0)));//结果列表
		ofstream out(resultfile);
		uniform_real_distribution<double> o(-0.0001, 0.0001);
		for (int f = 0; f < files.size(); f++) {//计算
			readdata(fs + files[f], isWeighted);
			cout << files[f] << endl;
			//topL = (int)(top_ratio*M);
			for (int t = 0; t < times; t++) {
				cout << t << " ";
				init(ratio);
				setalllink();
				rank.insert(rank.end(), test.begin(), test.end());
				cout << "divided" << endl;
				for (int m = 0; m < method.size(); m++) {
					rt = 0;
					for (int i = 0; i < rank.size(); i++) {//计算所有未连接边的score
						rank[i].score = sim(rank[i], method[m], 0) + o(e);
					}
					sort(rank.begin(), rank.end(), scoreGreater);//排序
					for (int i = 0; i < topL; i++) {
						if (rank[i].isTest) {//如果是测试边
							rt++;
						}
					}
					rt /= topL;
					res[f][m][t] = rt;
				}
				rank.clear();
			}
		}
		for (int f = 0; f < files.size(); f++) {//输出
			out << files[f] << endl;
			for (int m = 0; m < method.size(); m++) {
				for (int t = 0; t < times; t++) {
					out << res[f][m][t] << " ";
				}
				out << endl;
			}
			out << endl;
		}
		out.close();
	}

	void triesprecision2(int times, double top_ratio, double ratio, const char* path, string resultfile, bool isWeighted) {
		char path2[30];
		strcpy(path2, path);
		strcat(path2, "*.txt");
		getAllFile(path2);
		string fs = path;
		vector<int> method = { 1,2,3 };
		double rt = 0;
		int topL;
		vector<vector<vector<double> > > res(files.size(), vector<vector<double> >(method.size(), vector<double>(times, 0)));//结果列表
		ofstream out(resultfile);
		uniform_real_distribution<double> o(-0.0001, 0.0001);
		for (int f = 0; f < files.size(); f++) {//计算
			readdata(fs + files[f], isWeighted);
			cout << files[f] << endl;
			topL = (int)(top_ratio*M);
			for (int t = 0; t < times; t++) {
				cout << t << " ";
				init(ratio);
				setnolink(test.size());
				rank.clear();
				rank.insert(rank.end(), nolink.begin(), nolink.end());
				rank.insert(rank.end(), test.begin(), test.end());
				cout << "divided" << endl;
				for (int m = 0; m < method.size(); m++) {
					rt = 0;
					for (int i = 0; i < rank.size(); i++) {//计算所有未连接边的score
						rank[i].score = sim(rank[i], method[m], 0) + o(e);
					}
					sort(rank.begin(), rank.end(), scoreGreater);//排序
					for (int i = 0; i < topL; i++) {
						if (rank[i].isTest) {//如果是测试边
							rt++;
						}
					}
					rt /= topL;
					res[f][m][t] = rt;
				}
				rank.clear();
			}
		}
		for (int f = 0; f < files.size(); f++) {//输出
			out << files[f] << endl;
			for (int m = 0; m < method.size(); m++) {
				for (int t = 0; t < times; t++) {
					out << res[f][m][t] << " ";
				}
				out << endl;
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
		vector<int> method = { 10 };
		ofstream out(resultfile);
		for (int f = 0; f < files.size(); f++) {//遍历每个网络
			readdata(fs + files[f], true);
			out << files[f] << endl;
			cout << files[f] << endl;
			for (int m = 0; m < method.size(); m++) {//遍历每种算法
				cout << "method: " << method[m] << endl;
				a = -3;
				while (a <= 3) {//遍历alpha的取值
					result = 0;
					for (int t = 1; t <= times; t++) {
						init(ratio);
						switch (isAUC)
						{
						case 1:
							ai = AUC2(1000, method[m], a);
							out << ai << " ";
							break;
						case 0:
							//ai = Precision(method[m], 100, a);
							break;
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
						ai = AUC2(1000, 0, b);
						break;
					case 0:
						//ai = Precision(0, 100, b);
						break;
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
		double p11, p12, p21, p22;
		readdata(infile, false);
		init(0.1);
		ofstream out(outfile);
		for (int i = 0; i < test.size(); i++) {
			n11 = test[i].source;
			n12 = test[i].dest;
			n21 = nolink[i].source;
			n22 = nolink[i].dest;
			p11 = pn(n11, 1);
			p12 = pn(n12, 1);
			p21 = pn(n21, 1);
			p22 = pn(n22, 1);
			p11 = p11 / (p11 + p12);
			p12 = 1 - p11;
			p21 = p21 / (p21 + p22);
			p22 = 1 - p21;
			out << n11 << " " << n12 << " " << n21 << " " << n22 << " " << (double)1/(distance(n11, n12)-1) << " " << -p11 * log(p11) - p12 * log(p12) << " " << (double)1/(distance(n21, n22)-1) << " " << -p21 * log(p21) - p22 * log(p22) <<  endl;
		}
	}

	void showInfo2(string infile, string outfile) {
		int n11, n12, n21, n22;
		readdata(infile, false);
		init(0.1);
		ofstream out(outfile);
		for (int i = 0; i < test.size(); i++) {
			n11 = test[i].source;
			n12 = test[i].dest;
			n21 = nolink[i].source;
			n22 = nolink[i].dest;
			out << abs((int)(train[n11].size() - train[n12].size())) << " " << abs((int)(train[n21].size() - train[n22].size())) << endl;
		}
	}

};

int main(int argc, char **argv) {
	Network g;
	//g.readdata("F:/data/temp/USAir2.txt", false);
	g.triesAUC(1, 10000, true, 0.1, "F:/data/temp/", "F:/data/lp_data/result/923_AUC.txt", false);
	//g.triesprecision(10, 0.02, 0.1, "F:/data/temp/", "F:/data/lp_data/result/918_precision100_test2.txt", false);
	//g.triesprecision2(10, 0.02, 0.1, "F:/data/temp/", "F:/data/lp_data/result/918_precision2.txt", false);
	//g.showInfo("F:/data/temp/health.txt", "F:/data/info/health_info.txt");
	//g.showInfo2("F:/data/temp/health.txt", "F:/data/info/health_info2.txt");
	//g.tries2(10, 10000, true, 0.1, "F:/data/temp/", "F:/data/lp_data/result/912_AUC4.txt", false);
	system("pause");
}
