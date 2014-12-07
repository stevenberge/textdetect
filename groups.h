#ifndef GROUPS_H
#define GROUPS_H

float groupSim(vector<vector<float> > &graph,  vector<int> &group);

bool groupSco(RegionClassifier &, vector<Region> &, vector<int> &group);

float groupVar(vector<Region> &regions, vector<int> & group);



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

bool groupLs(vector<Region> &regions, vector<int> &group, int idx){
    float votes=0, sw_td=0, xl_sw = 0, n_hl = 0, n_r = 0, n_if = 0, n_ni =0, n_hw = 0;
    int N = group.size();
    float sw_var = 0;
    vector<float> xls(N, 0), wd(N, 0), hg(N, 0);
    for(int j = 0; j<group.size(); j++){
        int i = group[j];
        Region &r = regions[i];
        float xl= pow(pow(r.bbox_.height, 2)+pow(r.bbox_.width, 2), 0.5);
        //float vt= region_boost.get_votes(&r);
        //votes += vt;
        if(r.area_<0.85*r.bbox_.width*r.bbox_.height) n_ni++;
        if(r.area_>8) n_r++;
        //if(r.stroke_std_/r.stroke_mean_<0.4) sw_td++;
        //if(xl<30*r.stroke_mean_) xl_sw++;
         if(xl* r.stroke_mean_*1.35 < r.area_) xl_sw++;
        if(r.num_holes_>0 && r.num_holes_<3) n_hl++;
        if(r.bbox_.height < 3*r.bbox_.width && r.bbox_.height*1.5 >= r.bbox_.width) n_hw ++;
        //if(r.inflexion_num_>2) n_if ++;
        xls[j] = xl;
        wd[j] = r.bbox_.width, hg[j] = r.bbox_.height;
        sw_var += r.stroke_std_/(r.area_);
    }
    sw_td/=N, xl_sw/=N, n_hl/=N, n_r/=N, n_if/=N, n_ni/=N, n_hw/=N;
    sw_var/=N;
    cout<<"line"<<idx<<" sw_td:"<<sw_td<<" xl_sw:"<<xl_sw<<" n_hl:"<<n_hl<<" n_r:"<<n_r<<" n_if:"<<n_if<<" n_ni:"<<n_ni<<" n_hw:"<<n_hw<<endl;
    //if(sw_td<0.4) return 0;
    if(xl_sw<0.35) return 0;
    if(n_ni<0.35) return 0;
    if(n_hw<0.4) return 0;
    //if(n_hl<0.01) return 0;
    if(n_r<0.5 ) return 0;
    //if(n_if<0.4) return 0;
    if(sw_var>0.014) return 0;

    if(N>=5){
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
        if(hg_var>0.46) return false;
        if(wd_var>1) return false;
    }
    return true;
}


float groupVar(vector<Region> &regions, vector<int> & group){
  float stroke_mean=0, stroke_var=0, stroke_var1=0;
  for(int i=0; i<group.size(); i++){
    int k=group[i];
    stroke_mean+=regions[k].stroke_mean_;
  }
  stroke_mean/=group.size();
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





#endif // GROUPS_H
