#ifndef GROUPS_H
#define GROUPS_H

float groupSim(vector<vector<float> > &graph,  vector<int> &group);

bool groupSco(RegionClassifier &, vector<Region> &, vector<int> &group);

float groupVar(vector<Region> &regions, vector<int> & group);

struct cmp{
    bool operator ()(const Rect &a, const Rect &b){
        return a.x < b.x;
    }
}cmper;

float groupSim(vector<vector<float> > &graph, vector<int> &group){
    int N=group.size();
    vector<vector<float> > sims(N, vector<float>(N,0));
    float t=0;
    for(int i=0; i<N; i++){
        int pi = group[i];
        float ti=0;
        float tn=0;
        for(int j=0; j<N; j++){
            int pj = group[j];
            tn += graph[pi][pj];
        }
        //        ti=scores[i]/100;
        //        ti*=tn;
        t+=tn;
    }
    float s = t/N;//pow(N, 0.4);
    //cout<<"sum:"<<t<<" sim:"<<s<<endl;
    return s;
}

bool groupSco(RegionClassifier &region_boost, vector<Region> &regions, vector<int> &group){
    int N = group.size();
   // vector<int> tmp;

    int cnt = 0;
    for(int i=0; i<group.size(); i++){
      int k=group[i];
      if(regions[k].stroke_std_/regions[k].stroke_mean_<=0.05) cnt++;
    }
    return cnt>=N/2;
}
struct LineFeature{
    int n = 0;
    float sw_var;
    float n_sh;// height/stroke_mean
    float n_hw;//
    float n_nd;//cnt of non-dot
    float n_lr;//
    float xl_sw ;
    LineFeature(){
        sw_var =0, n_sh = 0, n_hw = 0, n_nd = 0, n_lr = 0, xl_sw = 0;
    }
};

bool groupLs(vector<Region> &regions, vector<int> &group, int idx, LineFeature &lf){
    float sw_td=0, xl_sw = 0, n_hl = 0, n_lr = 0, n_if = 0, n_nd =0, n_hw = 0, n_sh = 0;
    int N = group.size();
    float sw_var = 0;
    vector<float> xls(N, 0), wd(N, 0), hg(N, 0);
    for(int j = 0; j<group.size(); j++){
        int i = group[j];
        Region &r = regions[i];
        float xl= pow(pow(r.bbox_.height, 2)+pow(r.bbox_.width, 2), 0.5);
        //float vt= region_boost.get_votes(&r);
        //votes += vt;
//        if(r.area_<0.4*3.14/4*xl*xl) n_nd++;
        if(r.area_>10) n_lr++;
        if(!isDotStroke(r)) n_nd++;
        //if(r.stroke_std_/r.stroke_mean_<0.4) sw_td++;
        //if(xl<30*r.stroke_mean_) xl_sw++;
        if(r.bbox_.height>6* r.stroke_mean_) n_sh++;
         if(xl* r.stroke_mean_*1.75 < r.area_) xl_sw++;
        if(r.num_holes_>0 && r.num_holes_<3) n_hl++;
        if(r.bbox_.height < 3*r.bbox_.width && r.bbox_.height*1.5 >= r.bbox_.width) n_hw ++;
        //if(r.inflexion_num_>2) n_if ++;
        xls[j] = xl;
        wd[j] = r.bbox_.width, hg[j] = r.bbox_.height;
        sw_var += r.stroke_std_/(r.area_);
    }
    sw_td/=N, xl_sw/=N, n_hl/=N, n_lr/=N, n_if/=N, n_nd/=N, n_hw/=N, sw_var/=N;
    lf.n = N, lf.n_hw = n_hw, lf.n_nd = n_nd, lf.n_lr = n_lr, lf.n_sh = n_sh, lf.xl_sw=xl_sw, lf.sw_var=sw_var;
    cout<<"line"<<idx<<" sw_td:"<<sw_td<<" xl_sw:"<<xl_sw<<" n_hl:"<<n_hl<<" n_lr:"<<n_lr<<" n_if:"<<n_if<<" n_nd:"<<n_nd<<" n_hw:"<<n_hw<<endl;
    //if(sw_td<0.4) return 0;
    if(xl_sw<0.35) return 0;
    if(n_nd<0.35) return 0;
    if(n_hw<0.4) return 0;
    //if(n_hl<0.01) return 0;
    if(n_lr<0.3 ) return 0;
    if(n_sh<0.6) return 0;
    //if(n_if<0.4) return 0;

    if(sw_var>0.014) return 0;

    if(N>=2){
        float hg_var = 0, hg_mean = 0, wd_var = 0, wd_mean = 0;
        for(int i = 0;i<N; i++) hg_mean+=hg[i], wd_mean+=wd[i];
        hg_mean /= N;
        for(int i = 0;i<N; i++)
            hg_var += pow(hg[i] - hg_mean, 2),
            wd_var += pow(wd[i] - wd_mean, 2);
        hg_var /= pow(hg_mean, 2) * N;
        wd_var /= pow(wd_mean, 2) * N;
        hg_var = pow(hg_var, 0.5);
        wd_var = pow(wd_var, 0.5);
        cout<<"hg_var:"<<hg_var<<" wd_var:"<<wd_var<<endl;
        //////////////i don't know if this is good:if(hg_var>0.4) return false;
        if(hg_var>0.5) return false;
        if(wd_var>1) return false;
    }
    return true;
}

float groupSW(vector<Region> &regions, vector<int> & group){
    float stroke_mean = 0;
    for(int i=0; i<group.size(); i++){
      int k=group[i];
      stroke_mean+=regions[k].stroke_mean_;
    }
    stroke_mean/=group.size();
    return stroke_mean;
}

float groupVar(vector<Region> &regions, vector<int> & group){
  float stroke_mean=groupSW(regions, group), stroke_var=0, stroke_var1=0;
  //cout<<"stroke_mean:"<<stroke_mean<<endl;
  for(int i=0; i<group.size(); i++){
    int k=group[i];
    stroke_var+=regions[k].stroke_std_/regions[k].stroke_mean_;
    stroke_var1+=pow(regions[k].stroke_mean_-stroke_mean, 2);
  }
  stroke_var1 = pow(stroke_var1, 0.5);
  stroke_var1/= stroke_mean;//pow(stroke_var1, 0.5);
  return (stroke_var1*stroke_var)/group.size();///(stroke_mean*group.size());
}


pair<float, float> groupAlias(vector<Region> &regions, vector<int> &group){
    vector<Rect> rs;
    int n = group.size();
    assert(n>1);
    float tw = 0, th = 0;
    for(int i = 0; i<group.size(); i++)
        rs.push_back(regions[group[i]].bbox_),
        tw+= regions[group[i]].bbox_.width,
        th+= regions[group[i]].bbox_.height;
    tw/=n, th/=n;
    sort(rs.begin(), rs.end(), cmper);
    float offsety = 1;
    float offsetx = 1;
    Rect & r = rs[rs.size()-1];
    if(r.x == rs[0].x) return pair<float, float>(1, 1);
    float dx = (r.x - rs[0].x), dy = (r.y+r.height)-(rs[0].y+rs[0].height);
    if(dx==0) dx = 0.1;
    if(rs.size()==2){
        float oy = abs(dy)/th, ox = dx>2*tw? dx/tw-2: 0;
        return pair<float, float>(oy, ox);
    }
    for(int i =2; i<rs.size(); i++){\
        float dx= rs[i-1].x - rs[i-2].x;
        if(dx==0) dx = 0.1;
        float k = ((rs[i-1].y+rs[i-1].height)-(rs[i-2].y+rs[i-2].height))/dx;
        k = k>-100?k:-100, k = k<100?k:100;
        offsety += abs ( k*(rs[i].x - rs[i-1].x) - ((rs[i].y+rs[i].height) -(rs[i-1].y+rs[i-1].height)) );
        offsetx += abs((rs[i].x - rs[i-1].x) - dx);
    }
    return pair<float,float>(offsety/th, offsetx/tw);
}


#endif // GROUPS_H
