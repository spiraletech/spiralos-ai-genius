#include "spiral/scale.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
spiral::vision::RgbImage solid(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    spiral::vision::RgbImage image(4, 4);
    for (std::size_t y = 0; y < 4; ++y) for (std::size_t x = 0; x < 4; ++x) {
        image.at(x, y, 0) = r; image.at(x, y, 1) = g; image.at(x, y, 2) = b;
    }
    return image;
}
spiral::vision::RgbImage quadrants() {
    spiral::vision::RgbImage image(4, 4);
    for (std::size_t y = 0; y < 4; ++y) for (std::size_t x = 0; x < 4; ++x) {
        if (y < 2 && x < 2) image.at(x,y,0)=255;
        else if (y < 2) image.at(x,y,1)=255;
        else if (x < 2) image.at(x,y,2)=255;
        else { image.at(x,y,0)=255; image.at(x,y,1)=255; }
    }
    return image;
}
float distance(const spiral::Tensor& a, const spiral::Tensor& b) {
    float sum = 0.0F; for (std::size_t i=0;i<a.numel();++i) sum += std::abs(a.data()[i]-b.data()[i]); return sum;
}
}

int main() {
    using namespace spiral;
    using namespace spiral::multimodal;
    using namespace spiral::scale;

    const auto root = std::filesystem::temp_directory_path() / "spiral_l12";
    std::filesystem::remove_all(root); std::filesystem::create_directories(root);
    const auto red=solid(255,0,0), blue=solid(0,0,255), green=solid(0,255,0), yellow=solid(255,255,0);
    red.save_ppm((root/"r.ppm").string()); blue.save_ppm((root/"b.ppm").string());
    green.save_ppm((root/"g.ppm").string()); yellow.save_ppm((root/"y.ppm").string());
    { std::ofstream f(root/"data.tsv"); f << "r.ppm\tred\n" << "b.ppm\tblue\n" << "g.ppm\tgreen\n" << "y.ppm\tyellow\n"; }
    const auto dataset = load_manifest_tsv((root/"data.tsv").string());
    assert(dataset.size()==4); assert(shuffled_indices(9,77)==shuffled_indices(9,77));
    auto split = split_dataset(dataset,0.25F,99); assert(split.train.size()==3 && split.validation.size()==1);

    Random arng(101); AutoencoderConfig ac; ac.patch_size=2; ac.latent_dim=6; ImageAutoencoder ae(ac,arng);
    ImageTrainerConfig aetc; aetc.optimizer.learning_rate=0.03F; aetc.optimizer.weight_decay=0.0F; aetc.max_grad_norm=10.0F;
    AutoencoderTrainer aet(ae,aetc); const auto q=quadrants(); for(int i=0;i<350;++i)(void)aet.train_step(q);
    assert(aet.evaluate(red)<0.03F && aet.evaluate(blue)<0.03F);

    StableLatentTransformerConfig mc; mc.latent_dim=6; mc.model_dim=12; mc.num_heads=3; mc.num_layers=4;
    mc.ffn_dim=24; mc.text_feature_dim=12; mc.prompt_tokens=3; mc.time_feature_dim=4;
    Random mrng(202); StableLatentTransformerDenoiser model(mc,mrng); assert(std::abs(model.effective_residual_scale()-0.5F)<1e-6F);
    Random nrng(303); const Tensor noise=flow::gaussian_noise({4,6},nrng);
    const Tensor pr=model.predict(noise,"red",0.7F,2,2), pb=model.predict(noise,"blue",0.7F,2,2);
    assert(pr.shape()==std::vector<std::size_t>({4,6}) && distance(pr,pb)>1e-5F);

    ScaleTrainerConfig tc; tc.optimizer.learning_rate=0.004F; tc.optimizer.weight_decay=0.0F; tc.max_grad_norm=5.0F;
    tc.micro_batch_size=1; tc.gradient_accumulation_steps=2; tc.shuffle_seed=404; tc.noise_seed=505;
    ScaleTrainer trainer(model,ae,flow::NoiseScheduler{},tc);
    const float before=trainer.evaluate_dataset(split.train,606); ScaleMetrics m;
    for(int e=0;e<60;++e)m=trainer.train_epoch(split.train,split.validation);
    const float after=trainer.evaluate_dataset(split.train,606);
    assert(std::isfinite(after) && after<before*0.60F); assert(m.epoch==60 && m.optimizer_steps==120);

    flow::IterativeImageGenerator gen(ae,model); flow::SamplingConfig sc; sc.steps=3; sc.seed=707;
    assert(gen.generate("red",2,2,sc).pixels()==gen.generate("red",2,2,sc).pixels());
    append_metrics_csv((root/"metrics.csv").string(),m);

    save_training_checkpoint(model,trainer,(root/"resume.bin").string());
    Random crng(808); StableLatentTransformerDenoiser clone(mc,crng); ScaleTrainer clone_trainer(clone,ae,flow::NoiseScheduler{},tc);
    load_training_checkpoint(clone,clone_trainer,(root/"resume.bin").string());
    assert(clone_trainer.epoch()==trainer.epoch()); assert(clone_trainer.optimizer().step_count()==trainer.optimizer().step_count());
    assert(model.predict(noise,"resume",0.5F,2,2).data()==clone.predict(noise,"resume",0.5F,2,2).data());
    const auto a=trainer.train_epoch(split.train,split.validation), b=clone_trainer.train_epoch(split.train,split.validation);
    assert(a.train_loss==b.train_loss && a.validation_loss==b.validation_loss && a.last_grad_norm==b.last_grad_norm);
    const auto pa=model.parameters(), pc=clone.parameters(); assert(pa.size()==pc.size());
    for(std::size_t i=0;i<pa.size();++i) assert(pa[i]->value.data()==pc[i]->value.data());

    std::filesystem::remove_all(root);
    std::cout << "Spiral L12 scale tests passed\n" << before << " -> " << after << '\n';
    return 0;
}
